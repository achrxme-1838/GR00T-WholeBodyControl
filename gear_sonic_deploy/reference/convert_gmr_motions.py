#!/usr/bin/env python3
"""
Convert GMR retargeted motion .pkl files into the CSV directory layout the
C++ deploy binary expects (same as reference/example/<motion>/...).

GMR pkl schema (produced by General Motion Retargeting for G1 29DOF):
    fps              : float
    root_pos         : (T, 3)    world translation
    root_rot         : (T, 4)    xyzw quaternion
    dof_pos          : (T, 29)   joint angles, MuJoCo joint order
    local_body_pos   : (T, 38, 3) [unused here]
    link_body_list   : list[str] [reference only]

Output CSV layout (per motion):
    joint_pos.csv, joint_vel.csv      in IsaacLab joint order (29 dof)
    body_pos.csv, body_quat.csv       33 bodies in world frame
    body_lin_vel.csv, body_ang_vel.csv (finite-diff + Gaussian filter)
    metadata.txt, info.txt

The 33-body layout matches the deploy FK order (TRACKED_BODY_NAMES in
g1_29dof_sonic_distill.py) + 3 extended bodies:
    FK bodies 0..29     : MuJoCo body indices 1..30
                          (pelvis + left leg(6) + right leg(6) + waist+torso(3)
                           + left arm(7) + right arm(7) = 30)
    extended body 30    : left_hand_link_ext  (parent=left_wrist_yaw_link  [22],
                                               offset=(0.0415,  0.003, 0.0))
    extended body 31    : right_hand_link_ext (parent=right_wrist_yaw_link [29],
                                               offset=(0.0415, -0.003, 0.0))
    extended body 32    : head_link_ext       (parent=torso_link           [15],
                                               offset=(0.0,     0.0,  0.4))

Deploy uses `_body_indexes[j] == i` to locate FK body `i` (0..29) or extended
body `NUM_FK_BODIES + e` (30..32) in the motion CSV; writing an identity
`_body_indexes = [0, 1, ..., 32]` satisfies this lookup for all 33 bodies.

Usage:
    python3 convert_gmr_motions.py <pkl_file_or_dir> [output_dir] [--fps 50]
"""

import argparse
import os
import sys

import numpy as np

# 30 FK bodies in MuJoCo body index order. Verified via mj_id2name that
# MJ body 1..30 == TRACKED_BODY_NAMES[0..29] exactly.
NUM_FK_BODIES = 30
NUM_EXTENDED_BODIES = 3
NUM_TOTAL_BODIES = NUM_FK_BODIES + NUM_EXTENDED_BODIES  # 33

MJ_BODY_IDX_30 = np.arange(1, NUM_FK_BODIES + 1, dtype=np.int64)

# Extended body definitions: (parent_fk_idx, offset_xyz) — must match
# extended_body_defs_ in g1_deploy_onnx_ref.cpp and SMPLCfg.extending.extended_joints.
EXTENDED_BODY_DEFS = [
    (22, np.array([0.0415,  0.003, 0.0], dtype=np.float64)),   # left_hand_link_ext
    (29, np.array([0.0415, -0.003, 0.0], dtype=np.float64)),   # right_hand_link_ext
    (15, np.array([0.0,     0.0,   0.4], dtype=np.float64)),   # head_link_ext
]

# `_body_indexes` written to metadata is positional in the tracked+extended
# order, so deploy's `motion_body_indexes[j] == i` lookup is an identity map.
BODY_INDEX_33 = np.arange(NUM_TOTAL_BODIES, dtype=np.int64)

# Mapping between IsaacLab joint order and the motion-CSV ("Unitree SDK /
# mujoco-interleaved") order expected by the C++ deploy binary. Must mirror
# policy_parameters.hpp.
#
# The CSV joint_pos / joint_vel columns are NOT in chain order. The deploy
# reorders them via `reordered[j_il] = csv[isaaclab_to_mujoco[j_il]]` so the
# policy receives IsaacLab-chain order. GMR pkl's `dof_pos` is in chain order
# (matches mjcf joint traversal order, identical to IsaacLab chain order),
# so we permute as: `csv[k] = dof_pos_chain[mujoco_to_isaaclab[k]]`.
ISAACLAB_TO_MUJOCO = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28],
    dtype=np.int64,
)
MUJOCO_TO_ISAACLAB = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int64,
)


def _load_pkl(pkl_path):
    try:
        import joblib
        return joblib.load(pkl_path)
    except Exception:
        import pickle
        with open(pkl_path, "rb") as f:
            return pickle.load(f)


def _is_gmr_dict(d):
    return isinstance(d, dict) and {"root_pos", "root_rot", "dof_pos", "fps"}.issubset(d.keys())


def _normalize_pkl(data, default_name):
    """Return list of (motion_name, gmr_dict)."""
    if _is_gmr_dict(data):
        return [(default_name, data)]
    if isinstance(data, dict):
        motions = []
        for k, v in data.items():
            if _is_gmr_dict(v):
                motions.append((str(k), v))
        if motions:
            return motions
    raise ValueError("Input pkl is not a recognized GMR motion (expected keys: root_pos, root_rot, dof_pos, fps).")


def _quat_xyzw_to_wxyz(q):
    return np.stack([q[..., 3], q[..., 0], q[..., 1], q[..., 2]], axis=-1)


def _slerp(q0, q1, t):
    """SLERP between wxyz quaternions, arrays shape (..., 4), t shape (...,)."""
    q0 = q0 / (np.linalg.norm(q0, axis=-1, keepdims=True) + 1e-12)
    q1 = q1 / (np.linalg.norm(q1, axis=-1, keepdims=True) + 1e-12)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot)
    # Linear-blend fallback for nearly-parallel quaternions
    linear = dot > 0.9995
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    s0 = np.where(linear, 1.0 - t[..., None], np.sin((1.0 - t[..., None]) * theta) / (sin_theta + 1e-12))
    s1 = np.where(linear, t[..., None], np.sin(t[..., None] * theta) / (sin_theta + 1e-12))
    out = s0 * q0 + s1 * q1
    return out / (np.linalg.norm(out, axis=-1, keepdims=True) + 1e-12)


def _resample_linear(arr, src_fps, dst_fps):
    """Linearly resample (T, ...) array from src_fps to dst_fps."""
    if abs(src_fps - dst_fps) < 1e-6:
        return arr.copy()
    T = arr.shape[0]
    duration = (T - 1) / src_fps
    new_T = int(round(duration * dst_fps)) + 1
    src_t = np.linspace(0.0, duration, T)
    dst_t = np.linspace(0.0, duration, new_T)
    # numpy.interp is 1-D; loop over flat dims
    flat = arr.reshape(T, -1)
    out = np.empty((new_T, flat.shape[1]), dtype=arr.dtype)
    for i in range(flat.shape[1]):
        out[:, i] = np.interp(dst_t, src_t, flat[:, i])
    return out.reshape((new_T,) + arr.shape[1:])


def _resample_quat_slerp(q_wxyz, src_fps, dst_fps):
    """SLERP resample a (T, 4) wxyz quaternion trajectory."""
    if abs(src_fps - dst_fps) < 1e-6:
        return q_wxyz.copy()
    T = q_wxyz.shape[0]
    duration = (T - 1) / src_fps
    new_T = int(round(duration * dst_fps)) + 1
    dst_t = np.linspace(0.0, duration, new_T)
    src_t = np.linspace(0.0, duration, T)
    idx_float = np.interp(dst_t, src_t, np.arange(T))
    i0 = np.clip(np.floor(idx_float).astype(int), 0, T - 1)
    i1 = np.clip(i0 + 1, 0, T - 1)
    frac = idx_float - i0
    return _slerp(q_wxyz[i0], q_wxyz[i1], frac)


def _compute_world_fk(model, data, root_pos, root_quat_wxyz, dof_pos_mj, mj_body_idx):
    """Run mj_forward per frame, return body_pos (T, K, 3) and body_quat_wxyz (T, K, 4)."""
    import mujoco

    T = root_pos.shape[0]
    K = len(mj_body_idx)
    body_pos = np.zeros((T, K, 3), dtype=np.float64)
    body_quat = np.zeros((T, K, 4), dtype=np.float64)

    for t in range(T):
        # MuJoCo qpos: [root_pos(3), root_quat_wxyz(4), dof(29)]
        data.qpos[0:3] = root_pos[t]
        data.qpos[3:7] = root_quat_wxyz[t]
        data.qpos[7:7 + 29] = dof_pos_mj[t]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for k, bi in enumerate(mj_body_idx):
            body_pos[t, k] = data.xpos[bi]
            body_quat[t, k] = data.xquat[bi]  # MuJoCo xquat is wxyz

    return body_pos, body_quat


def _append_extended_bodies(body_pos_fk, body_quat_fk):
    """Append the 3 extended bodies to FK arrays.

    Extended body pose:
        pos = parent_pos + quat_rotate(parent_quat, offset)
        rot = parent_rot  (child shares parent rotation, matches deploy
              ComputeExtendedBodyRot)
    """
    T, K_fk, _ = body_pos_fk.shape
    assert K_fk == NUM_FK_BODIES, f"expected {NUM_FK_BODIES} FK bodies, got {K_fk}"

    body_pos = np.zeros((T, NUM_TOTAL_BODIES, 3), dtype=np.float64)
    body_quat = np.zeros((T, NUM_TOTAL_BODIES, 4), dtype=np.float64)
    body_pos[:, :K_fk] = body_pos_fk
    body_quat[:, :K_fk] = body_quat_fk

    for e, (parent_idx, offset) in enumerate(EXTENDED_BODY_DEFS):
        parent_pos = body_pos_fk[:, parent_idx]          # (T, 3)
        parent_quat = body_quat_fk[:, parent_idx]        # (T, 4) wxyz
        # Rotate local offset by parent quaternion (world-frame offset)
        offset_rot = _quat_rotate_wxyz(parent_quat, np.broadcast_to(offset, parent_pos.shape))
        body_pos[:, K_fk + e] = parent_pos + offset_rot
        body_quat[:, K_fk + e] = parent_quat

    return body_pos, body_quat


def _quat_rotate_wxyz(q, v):
    """Rotate vectors v by quaternions q (wxyz). Shapes: q (..., 4), v (..., 3)."""
    w = q[..., 0:1]
    xyz = q[..., 1:4]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def _quat_mul_wxyz(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def _quat_conj_wxyz(q):
    out = q.copy()
    out[..., 1:] = -out[..., 1:]
    return out


def _quat_to_angle_axis(q):
    """q: (..., 4) wxyz. Returns angle (...,) and axis (..., 3)."""
    w = np.clip(q[..., 0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(np.clip(1.0 - w * w, 0.0, 1.0))
    safe = sin_half > 1e-6
    axis = np.zeros_like(q[..., 1:])
    axis[safe] = q[..., 1:][safe] / sin_half[safe][..., None]
    axis[~safe] = np.array([1.0, 0.0, 0.0])
    # Wrap angle to (-pi, pi]
    angle = np.where(angle > np.pi, angle - 2.0 * np.pi, angle)
    return angle, axis


def _gaussian_kernel(sigma=2.0):
    lw = int(4.0 * sigma + 0.5)
    dx = np.arange(-lw, lw + 1)
    k = np.exp(-0.5 * (dx / sigma) ** 2)
    return k / k.sum()


def _gaussian_filter_nearest(x, sigma=2.0, axis=0):
    """Replicate scipy.ndimage.gaussian_filter1d(mode='nearest') without scipy."""
    k = _gaussian_kernel(sigma)
    lw = len(k) // 2
    pad = [(0, 0)] * x.ndim
    pad[axis] = (lw, lw)
    xp = np.pad(x, pad, mode="edge")
    out = np.zeros_like(x, dtype=np.float64)
    for i, w in enumerate(k):
        sl_src = [slice(None)] * x.ndim
        sl_src[axis] = slice(i, i + x.shape[axis])
        out += w * xp[tuple(sl_src)]
    return out


def _compute_body_velocities(body_pos, body_quat_wxyz, fps, filter=True):
    """Finite-diff linear/angular velocities matching C++ ComputeGlobalVelocities.
       body_pos: (T, K, 3), body_quat_wxyz: (T, K, 4)."""
    T = body_pos.shape[0]
    lin = np.zeros_like(body_pos)
    ang = np.zeros_like(body_pos)
    for f in range(T):
        f0 = max(f - 1, 0)
        f1 = min(T - 1, f + 1)
        dt_frames = max(1, f1 - f0)
        dt = dt_frames / fps
        lin[f] = (body_pos[f1] - body_pos[f0]) / dt
        q1 = body_quat_wxyz[f1]
        q0 = body_quat_wxyz[max(0, f1 - 1)]
        dq = _quat_mul_wxyz(q1, _quat_conj_wxyz(q0))
        angle, axis = _quat_to_angle_axis(dq)
        ang[f] = axis * angle[..., None] * fps
    if filter:
        lin = _gaussian_filter_nearest(lin, sigma=2.0, axis=0)
        ang = _gaussian_filter_nearest(ang, sigma=2.0, axis=0)
    return lin, ang


def _save_csv(path, array, headers):
    with open(path, "w") as f:
        f.write(",".join(headers) + "\n")
        for row in array:
            f.write(",".join(f"{v:.6f}" for v in row) + "\n")


def _write_metadata(path, motion_name, T, body_indexes):
    K = len(body_indexes)
    # The deploy metadata reader consumes exactly one line after
    # "Body part indexes:" and regex-matches \d+ tokens, so the whole array
    # MUST fit on a single line. np.array2string wraps by default at ~75 cols
    # (which silently truncated bodies 24+ on 33-body metadata).
    body_idx_str = np.array2string(
        np.asarray(body_indexes),
        separator=" ",
        max_line_width=10_000,
        threshold=10_000,
    )
    with open(path, "w") as f:
        f.write(f"Metadata for: {motion_name}\n")
        f.write("=" * 30 + "\n\n")
        f.write("Body part indexes:\n")
        f.write(f"{body_idx_str}\n\n")
        f.write(f"Total timesteps: {T}\n\n")
        f.write("Data arrays summary:\n")
        f.write(f"  joint_pos: ({T}, 29) (float32)\n")
        f.write(f"  joint_vel: ({T}, 29) (float32)\n")
        f.write(f"  body_pos_w: ({T}, {K}, 3) (float32)\n")
        f.write(f"  body_quat_w: ({T}, {K}, 4) (float32)\n")
        f.write(f"  body_lin_vel_w: ({T}, {K}, 3) (float32)\n")
        f.write(f"  body_ang_vel_w: ({T}, {K}, 3) (float32)\n")
        f.write(f"  _body_indexes: ({K},) (int64)\n")
        f.write(f"  time_step_total: () (int64)\n")


def _write_info(path, motion_name, arrays):
    with open(path, "w") as f:
        f.write(f"Motion Information: {motion_name}\n")
        f.write("=" * 50 + "\n\n")
        for key, arr in arrays.items():
            f.write(f"{key}:\n")
            f.write(f"  Shape: {arr.shape}\n")
            f.write(f"  Dtype: {arr.dtype}\n")
            if arr.size > 0:
                flat = arr.flatten()
                f.write(f"  Range: [{float(flat.min()):.3f}, {float(flat.max()):.3f}]\n")
                f.write(f"  Sample: {flat[:5]}\n")
            f.write("\n")


def convert_one_motion(motion_name, gmr, output_dir, model, data, target_fps, filter_vel=True):
    os.makedirs(output_dir, exist_ok=True)

    src_fps = float(gmr["fps"])
    root_pos = np.asarray(gmr["root_pos"], dtype=np.float64)
    root_rot_xyzw = np.asarray(gmr["root_rot"], dtype=np.float64)
    dof_pos_mj = np.asarray(gmr["dof_pos"], dtype=np.float64)
    if dof_pos_mj.ndim == 3 and dof_pos_mj.shape[-1] == 1:
        dof_pos_mj = dof_pos_mj.squeeze(-1)

    root_rot_wxyz = _quat_xyzw_to_wxyz(root_rot_xyzw)
    root_rot_wxyz /= (np.linalg.norm(root_rot_wxyz, axis=-1, keepdims=True) + 1e-12)

    if target_fps is not None and abs(target_fps - src_fps) > 1e-6:
        root_pos = _resample_linear(root_pos, src_fps, target_fps)
        dof_pos_mj = _resample_linear(dof_pos_mj, src_fps, target_fps)
        root_rot_wxyz = _resample_quat_slerp(root_rot_wxyz, src_fps, target_fps)
        eff_fps = float(target_fps)
    else:
        eff_fps = src_fps

    T = root_pos.shape[0]

    # Forward kinematics -> world body pose for 30 FK bodies,
    # then append 3 extended bodies (parent pose + rotated offset).
    body_pos_fk, body_quat_fk = _compute_world_fk(
        model, data, root_pos, root_rot_wxyz, dof_pos_mj, MJ_BODY_IDX_30
    )
    body_pos, body_quat_wxyz = _append_extended_bodies(body_pos_fk, body_quat_fk)

    # Finite-diff body velocities (matches C++ ComputeGlobalVelocities).
    # Computed on the extended 33-body arrays so extended bodies get proper
    # velocities from the parent+offset trajectory.
    body_lin_vel, body_ang_vel = _compute_body_velocities(body_pos, body_quat_wxyz, eff_fps, filter=filter_vel)

    # GMR pkl's dof_pos is in chain order == IsaacLab joint order.
    # CSV must be in "mujoco-interleaved" order (what the C++ deploy expects
    # before applying its own isaaclab_to_mujoco reorder). Permute accordingly.
    joint_pos_csv = dof_pos_mj[:, MUJOCO_TO_ISAACLAB]

    # Joint velocity: finite difference of chain-order dof, then same permutation
    dt = 1.0 / eff_fps
    if T >= 2:
        dv_chain = (dof_pos_mj[1:] - dof_pos_mj[:-1]) / dt
        joint_vel_chain = np.concatenate([dv_chain, dv_chain[-1:]], axis=0)
    else:
        joint_vel_chain = np.zeros_like(dof_pos_mj)
    joint_vel_csv = joint_vel_chain[:, MUJOCO_TO_ISAACLAB]

    # Cast to float32 for CSV round-trip consistency with existing example format
    joint_pos_csv = joint_pos_csv.astype(np.float32)
    joint_vel_csv = joint_vel_csv.astype(np.float32)
    body_pos = body_pos.astype(np.float32)
    body_quat_wxyz = body_quat_wxyz.astype(np.float32)
    body_lin_vel = body_lin_vel.astype(np.float32)
    body_ang_vel = body_ang_vel.astype(np.float32)

    # Write CSVs
    n_joints = joint_pos_csv.shape[1]
    _save_csv(os.path.join(output_dir, "joint_pos.csv"), joint_pos_csv,
              [f"joint_{i}" for i in range(n_joints)])
    _save_csv(os.path.join(output_dir, "joint_vel.csv"), joint_vel_csv,
              [f"joint_vel_{i}" for i in range(n_joints)])

    K = body_pos.shape[1]
    _save_csv(os.path.join(output_dir, "body_pos.csv"), body_pos.reshape(T, -1),
              [f"body_{b}_{c}" for b in range(K) for c in "xyz"])
    _save_csv(os.path.join(output_dir, "body_quat.csv"), body_quat_wxyz.reshape(T, -1),
              [f"body_{b}_{c}" for b in range(K) for c in "wxyz"])
    _save_csv(os.path.join(output_dir, "body_lin_vel.csv"), body_lin_vel.reshape(T, -1),
              [f"body_{b}_vel_{c}" for b in range(K) for c in "xyz"])
    _save_csv(os.path.join(output_dir, "body_ang_vel.csv"), body_ang_vel.reshape(T, -1),
              [f"body_{b}_angvel_{c}" for b in range(K) for c in "xyz"])

    _write_metadata(os.path.join(output_dir, "metadata.txt"), motion_name, T, BODY_INDEX_33)
    _write_info(
        os.path.join(output_dir, "info.txt"),
        motion_name,
        {
            "joint_pos": joint_pos_csv,
            "joint_vel": joint_vel_csv,
            "body_pos_w": body_pos,
            "body_quat_w": body_quat_wxyz,
            "body_lin_vel_w": body_lin_vel,
            "body_ang_vel_w": body_ang_vel,
            "_body_indexes": BODY_INDEX_33,
        },
    )

    return T, eff_fps


def convert_pkl(pkl_path, output_base, mjcf_path, target_fps, filter_vel=True):
    import mujoco

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)
    assert model.nq == 36 and model.nbody == 31, (
        f"MJCF {mjcf_path} has nq={model.nq}, nbody={model.nbody}; expected nq=36, nbody=31 for g1_29dof.xml"
    )

    pkl_stem = os.path.splitext(os.path.basename(pkl_path))[0]
    pkl_data = _load_pkl(pkl_path)
    motions = _normalize_pkl(pkl_data, default_name=pkl_stem)

    os.makedirs(output_base, exist_ok=True)

    converted = []
    for motion_name, gmr in motions:
        safe_name = motion_name.replace("/", "_").replace(" ", "_")
        motion_dir = os.path.join(output_base, safe_name)
        T, fps = convert_one_motion(motion_name, gmr, motion_dir, model, data, target_fps, filter_vel=filter_vel)
        converted.append((motion_name, safe_name, T, fps))
        print(f"  [OK] {motion_name} -> {motion_dir} (T={T}, fps={fps:.1f})")

    return converted


def main():
    parser = argparse.ArgumentParser(description="Convert GMR .pkl motions to CSV directories.")
    parser.add_argument("input", help="GMR .pkl file or directory containing .pkl files")
    parser.add_argument("output_dir", nargs="?", default=None, help="Output base directory (default: <input>_csv)")
    parser.add_argument(
        "--mjcf",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "g1", "g1_29dof.xml"),
        help="Path to g1_29dof.xml (default: gear_sonic_deploy/g1/g1_29dof.xml)",
    )
    parser.add_argument("--fps", type=float, default=50.0, help="Target fps for output (default: 50). Use 0 to keep source fps.")
    parser.add_argument("--no-filter", action="store_true", help="Disable Gaussian filter on finite-diff velocities.")
    args = parser.parse_args()

    mjcf_path = os.path.abspath(args.mjcf)
    if not os.path.exists(mjcf_path):
        print(f"Error: MJCF not found: {mjcf_path}", file=sys.stderr)
        return 1

    target_fps = None if args.fps <= 0 else float(args.fps)

    inputs = []
    if os.path.isdir(args.input):
        for fname in sorted(os.listdir(args.input)):
            if fname.endswith(".pkl"):
                inputs.append(os.path.join(args.input, fname))
    elif os.path.isfile(args.input):
        inputs.append(args.input)
    else:
        print(f"Error: input not found: {args.input}", file=sys.stderr)
        return 1

    if not inputs:
        print(f"Error: no .pkl files found in {args.input}", file=sys.stderr)
        return 1

    if args.output_dir is None:
        if os.path.isdir(args.input):
            output_base = os.path.join(os.path.dirname(os.path.abspath(args.input)),
                                       os.path.basename(os.path.abspath(args.input).rstrip("/")) + "_csv")
        else:
            output_base = os.path.join(os.path.dirname(os.path.abspath(args.input)),
                                       os.path.splitext(os.path.basename(args.input))[0] + "_csv")
    else:
        output_base = os.path.abspath(args.output_dir)

    os.makedirs(output_base, exist_ok=True)

    print(f"MJCF        : {mjcf_path}")
    print(f"Target fps  : {target_fps if target_fps else 'keep source'}")
    print(f"Output base : {output_base}")
    print(f"Inputs      : {len(inputs)} pkl file(s)")
    print("")

    total = 0
    for pkl in inputs:
        print(f"-> {pkl}")
        converted = convert_pkl(pkl, output_base, mjcf_path, target_fps, filter_vel=not args.no_filter)
        total += len(converted)

    print(f"\nDone. Converted {total} motion(s) into {output_base}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
