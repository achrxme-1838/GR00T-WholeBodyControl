#!/bin/bash
set -e

# ============================================================================
# G1 Deploy (MoE Student) - Single-model policy deployment
# ============================================================================
# Deploys a single monolithic policy: obs -> policy.onnx -> action
# No encoder/decoder split. No token_state.
#
# Usage: ./deploy_moe_student.sh [OPTIONS] [sim|real|<interface_name>|<ip_address>]
#
# Default: real
# ============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================================================
# Interface Resolution Functions (same as deploy.sh)
# ============================================================================

get_network_interfaces() {
    if [[ "$(uname)" == "Darwin" ]]; then
        ifconfig | awk '
            /^[a-z]/ { iface=$1; gsub(/:$/, "", iface) }
            /inet / { print iface ":" $2 }
        '
    else
        ip -4 addr show 2>/dev/null | awk '
            /^[0-9]+:/ { gsub(/:$/, "", $2); iface=$2 }
            /inet / { split($2, a, "/"); print iface ":" a[1] }
        ' 2>/dev/null || \
        ifconfig 2>/dev/null | awk '
            /^[a-z]/ { iface=$1; gsub(/:$/, "", iface) }
            /inet / {
                for (i=1; i<=NF; i++) {
                    if ($i == "inet") { print iface ":" $(i+1); break }
                    if ($i ~ /^addr:/) { split($i, a, ":"); print iface ":" a[2]; break }
                }
            }
        '
    fi
}

find_interface_by_ip() {
    local target_ip="$1"
    get_network_interfaces | while IFS=: read -r iface ip; do
        if [[ "$ip" == "$target_ip" ]]; then
            echo "$iface"
            return 0
        fi
    done
}

find_interface_by_ip_prefix() {
    local prefix="$1"
    get_network_interfaces | while IFS=: read -r iface ip; do
        if [[ "$ip" == "$prefix"* ]]; then
            echo "$iface"
            return 0
        fi
    done
}

is_ip_address() {
    local input="$1"
    if [[ "$input" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        return 0
    fi
    return 1
}

interface_has_ip() {
    local iface="$1"
    local target_ip="$2"
    get_network_interfaces | while IFS=: read -r name ip; do
        if [[ "$name" == "$iface" ]] && [[ "$ip" == "$target_ip" ]]; then
            echo "yes"
            return 0
        fi
    done
}

resolve_interface() {
    local interface="$1"
    local os_type="$(uname)"

    if is_ip_address "$interface"; then
        if [[ "$interface" == "127.0.0.1" ]]; then
            TARGET="$interface"
            ENV_TYPE="sim"
        else
            TARGET="$interface"
            ENV_TYPE="real"
        fi
        return 0
    fi

    if [[ "$interface" == "sim" ]]; then
        local lo_interface
        lo_interface=$(find_interface_by_ip "127.0.0.1")

        if [[ -n "$lo_interface" ]]; then
            if [[ "$os_type" == "Darwin" ]] && [[ "$lo_interface" == "lo" ]]; then
                TARGET="lo0"
            else
                TARGET="$lo_interface"
            fi
        else
            if [[ "$os_type" == "Darwin" ]]; then
                TARGET="lo0"
            else
                TARGET="lo"
            fi
        fi
        ENV_TYPE="sim"
        return 0

    elif [[ "$interface" == "real" ]]; then
        local real_interface
        real_interface=$(find_interface_by_ip_prefix "192.168.123.")

        if [[ -n "$real_interface" ]]; then
            TARGET="$real_interface"
        else
            local fallback_interface
            fallback_interface=$(get_network_interfaces | grep -v "127.0.0.1" | head -1 | cut -d: -f1)

            if [[ -n "$fallback_interface" ]]; then
                TARGET="$fallback_interface"
                echo -e "${YELLOW}Could not find 192.168.123.x interface, using: $TARGET${NC}" >&2
            else
                TARGET="enP8p1s0"
                echo -e "${YELLOW}Could not auto-detect interface, using default: $TARGET${NC}" >&2
            fi
        fi
        ENV_TYPE="real"
        return 0

    else
        local has_loopback
        has_loopback=$(interface_has_ip "$interface" "127.0.0.1")

        if [[ "$has_loopback" == "yes" ]]; then
            TARGET="$interface"
            ENV_TYPE="sim"
            return 0
        fi

        if [[ "$os_type" == "Darwin" ]] && [[ "$interface" == "lo" ]]; then
            TARGET="lo0"
            ENV_TYPE="sim"
            return 0
        fi

        TARGET="$interface"
        ENV_TYPE="real"
        return 0
    fi
}

# ============================================================================
# Parse Command Line Arguments
# ============================================================================

show_usage() {
    echo "Usage: $0 [OPTIONS] [sim|real|<interface>]"
    echo ""
    echo "Deploy a single monolithic policy (obs -> policy.onnx -> action)."
    echo "No encoder/decoder split."
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  --policy PATH           Set the policy ONNX file (default: policy/moe_student/policy.onnx)"
    echo "  --obs-config PATH       Set the observation config file (default: policy/moe_student/observation_config.yaml)"
    echo "  --planner PATH          Set the planner model path (default: planner/target_vel/V2/planner_sonic.onnx)"
    echo "  --motion-data PATH      Motion data path. Accepts:"
    echo "                            - CSV motion directory (reference/example/)"
    echo "                            - GMR .pkl file or directory of .pkls (auto-converted)"
    echo "                          (default: $MOTION_DATA_DEFAULT)"
    echo "  --gmr-fps FPS           Target playback fps for GMR conversion (default: $GMR_FPS_DEFAULT; 0 = keep source)"
    echo "  --input-type TYPE       Set the input type (default: manager)"
    echo "  --output-type TYPE      Set the output type (default: all)"
    echo "  --zmq-host HOST         Set the ZMQ host (default: localhost)"
    echo ""
    echo "Interface modes:"
    echo "  sim              Use loopback interface for simulation (MuJoCo)"
    echo "  real             Auto-detect robot network (192.168.123.x)"
    echo "  <interface>      Use specific interface (e.g., enP8p1s0, eth0)"
    echo "  <ip_address>     Use interface by IP address"
    echo ""
    echo "Default: real"
    echo ""
    echo "Examples:"
    echo "  $0 sim"
    echo "  $0 --policy policy/moe_student/my_policy.onnx sim"
    echo "  $0 --policy path/to/policy.onnx --obs-config path/to/obs.yaml real"
}

# Default interface mode
INTERFACE_MODE="real"

# Default configuration values for MoE student (single model)
# POLICY_DEFAULT="policy/moe_student/policy.onnx"
# OBS_CONFIG_DEFAULT="policy/moe_student/observation_config.yaml"
# POLICY_DEFAULT="policy/moe_student_test/student.onnx"
# OBS_CONFIG_DEFAULT="policy/moe_student_test/observation_config.yaml"
# POLICY_DEFAULT="policy/moe_student_test_2nd/student.onnx"
# OBS_CONFIG_DEFAULT="policy/moe_student_test_2nd/observation_config.yaml"
# POLICY_DEFAULT="policy/moe_student_test_3rd/student.onnx"
# OBS_CONFIG_DEFAULT="policy/moe_student_test_3rd/observation_config.yaml"
# POLICY_DEFAULT="policy/moe_student_test_4th/student_push.onnx"
# OBS_CONFIG_DEFAULT="policy/moe_student_test_4th/observation_config.yaml"

POLICY_DEFAULT="policy/moe_student_test_6th/student.onnx"
OBS_CONFIG_DEFAULT="policy/moe_student_test_6th/observation_config.yaml"

PLANNER_DEFAULT="planner/target_vel/V2/planner_sonic.onnx"
# MOTION_DATA_DEFAULT="reference/example/"
# Accepts any of:
#   - a CSV motion directory (reference/example/)
#   - a directory of GMR .pkl files (reference/GMR/) -> auto-converted
#   - a single GMR .pkl file                          -> auto-converted
MOTION_DATA_DEFAULT="reference/GMR/"
GMR_CONVERTER_DEFAULT="reference/convert_gmr_motions.py"
GMR_OUTPUT_CACHE_DEFAULT="reference/_gmr_cache/"
GMR_FPS_DEFAULT="50"
INPUT_TYPE_DEFAULT="manager"
OUTPUT_TYPE_DEFAULT="all"
ZMQ_HOST_DEFAULT="localhost"

# Initialize with defaults
POLICY="$POLICY_DEFAULT"
OBS_CONFIG="$OBS_CONFIG_DEFAULT"
PLANNER="$PLANNER_DEFAULT"
MOTION_DATA="$MOTION_DATA_DEFAULT"
INPUT_TYPE="$INPUT_TYPE_DEFAULT"
OUTPUT_TYPE="$OUTPUT_TYPE_DEFAULT"
ZMQ_HOST="$ZMQ_HOST_DEFAULT"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        --policy)
            if [[ -z "$2" ]]; then
                echo -e "${RED}Error: --policy requires a path argument${NC}" >&2
                exit 1
            fi
            POLICY="$2"
            shift 2
            ;;
        --obs-config)
            if [[ -z "$2" ]]; then
                echo -e "${RED}Error: --obs-config requires a path argument${NC}" >&2
                exit 1
            fi
            OBS_CONFIG="$2"
            shift 2
            ;;
        --planner)
            if [[ -z "$2" ]]; then
                echo -e "${RED}Error: --planner requires a path argument${NC}" >&2
                exit 1
            fi
            PLANNER="$2"
            shift 2
            ;;
        --motion-data)
            if [[ -z "$2" ]]; then
                echo -e "${RED}Error: --motion-data requires a path argument${NC}" >&2
                exit 1
            fi
            MOTION_DATA="$2"
            shift 2
            ;;
        --gmr-fps)
            if [[ -z "$2" ]]; then
                echo -e "${RED}Error: --gmr-fps requires a value${NC}" >&2
                exit 1
            fi
            GMR_FPS_DEFAULT="$2"
            shift 2
            ;;
        --input-type)
            if [[ -z "$2" ]]; then
                echo -e "${RED}Error: --input-type requires a type argument${NC}" >&2
                exit 1
            fi
            INPUT_TYPE="$2"
            shift 2
            ;;
        --output-type)
            if [[ -z "$2" ]]; then
                echo -e "${RED}Error: --output-type requires a type argument${NC}" >&2
                exit 1
            fi
            OUTPUT_TYPE="$2"
            shift 2
            ;;
        --zmq-host)
            if [[ -z "$2" ]]; then
                echo -e "${RED}Error: --zmq-host requires a host argument${NC}" >&2
                exit 1
            fi
            ZMQ_HOST="$2"
            shift 2
            ;;
        sim|real)
            INTERFACE_MODE="$1"
            shift
            ;;
        *)
            INTERFACE_MODE="$1"
            shift
            ;;
    esac
done

# ============================================================================
# Display Header
# ============================================================================

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                  G1 DEPLOY - MOE STUDENT (SINGLE POLICY)             ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ============================================================================
# Resolve Interface
# ============================================================================

echo -e "${BLUE}[Interface Resolution]${NC}"
echo "Requested mode: $INTERFACE_MODE"

resolve_interface "$INTERFACE_MODE"

echo -e "Resolved interface: ${GREEN}$TARGET${NC}"
echo -e "Environment type:   ${GREEN}$ENV_TYPE${NC}"
echo ""

# ============================================================================
# Configuration
# ============================================================================

# Single policy model (no encoder/decoder split)
# The C++ binary takes argv[2] as the "model file" (originally decoder).
# We pass our single policy.onnx directly. No --encoder-file is passed.

# Additional flags for simulation mode
# --dex-rl-lab: enables IsaacLab joint order, uniform action scale 0.25,
#               DEX_RL_LAB default angles, and obs scaling (base_ang_vel*0.2, joint_vel*0.05)
EXTRA_ARGS="--dex-rl-lab"
if [[ "$ENV_TYPE" == "sim" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --disable-crc-check"
    echo -e "${YELLOW}Simulation mode: CRC check will be disabled${NC}"
    echo ""
fi

# ============================================================================
# Step 1: Check Prerequisites
# ============================================================================

echo -e "${BLUE}[Step 1/4]${NC} Checking prerequisites..."

# Check for TensorRT
if [ -z "$TensorRT_ROOT" ]; then
    echo -e "${YELLOW}TensorRT_ROOT is not set.${NC}"
    echo "   Please ensure TensorRT is installed and add to your ~/.bashrc:"
    echo "   export TensorRT_ROOT=\$HOME/TensorRT"

    if [ -d "$HOME/TensorRT" ]; then
        echo -e "${GREEN}   Found TensorRT at ~/TensorRT - setting temporarily${NC}"
        export TensorRT_ROOT="$HOME/TensorRT"
    fi
fi

# Check for required model files
check_file() {
    if [ ! -f "$1" ]; then
        echo -e "${RED}Missing file: $1${NC}"
        return 1
    else
        echo -e "${GREEN}Found: $1${NC}"
        return 0
    fi
}

echo ""
echo "Checking required model files..."
MISSING_FILES=0

check_file "$POLICY" || MISSING_FILES=$((MISSING_FILES + 1))
check_file "$OBS_CONFIG" || MISSING_FILES=$((MISSING_FILES + 1))
check_file "$PLANNER" || MISSING_FILES=$((MISSING_FILES + 1))

# Motion data can be:
#   - a CSV motion directory (contains subdirs with joint_pos.csv etc.)
#   - a directory of GMR .pkl files (requires conversion)
#   - a single GMR .pkl file (requires conversion)
dir_has_gmr_pkls() {
    local dir="$1"
    [ -d "$dir" ] && compgen -G "$dir"/*.pkl > /dev/null
}

dir_has_csv_motions() {
    local dir="$1"
    if [ ! -d "$dir" ]; then return 1; fi
    local sub
    for sub in "$dir"/*/; do
        if [ -f "$sub/joint_pos.csv" ] || [ -f "$sub/body_pos.csv" ]; then
            return 0
        fi
    done
    return 1
}

# Hash the source path + mtime so the cache key changes when source files do.
# Bump GMR_CSV_SCHEMA_VERSION whenever convert_gmr_motions.py changes its
# output layout (e.g. adding bodies) so stale caches are automatically
# invalidated without requiring manual cleanup.
GMR_CSV_SCHEMA_VERSION="v2-33body"
gmr_cache_key() {
    local src="$1"
    local stamp
    if [ -d "$src" ]; then
        stamp=$(find "$src" -maxdepth 1 -name '*.pkl' -printf '%f:%T@\n' 2>/dev/null | sort)
    else
        stamp=$(stat -c '%n:%Y' "$src" 2>/dev/null)
    fi
    printf '%s|%s' "$GMR_CSV_SCHEMA_VERSION" "$stamp" | sha1sum | awk '{print $1}' | cut -c1-12
}

convert_gmr_if_needed() {
    local src="$1"
    local is_pkl_source=false
    if [ -f "$src" ] && [[ "$src" == *.pkl ]]; then
        is_pkl_source=true
    elif dir_has_gmr_pkls "$src" && ! dir_has_csv_motions "$src"; then
        is_pkl_source=true
    fi

    if ! $is_pkl_source; then
        echo "$src"
        return 0
    fi

    if [ ! -f "$GMR_CONVERTER_DEFAULT" ]; then
        echo -e "${RED}GMR converter not found: $GMR_CONVERTER_DEFAULT${NC}" >&2
        return 1
    fi

    local key
    key=$(gmr_cache_key "$src")
    local src_stem
    if [ -f "$src" ]; then
        src_stem=$(basename "$src" .pkl)
    else
        src_stem=$(basename "$(cd "$src" && pwd)")
    fi
    local out_dir="$GMR_OUTPUT_CACHE_DEFAULT$src_stem-$key"

    if [ -d "$out_dir" ] && dir_has_csv_motions "$out_dir"; then
        echo -e "${GREEN}Using cached GMR conversion: $out_dir${NC}" >&2
    else
        echo -e "${YELLOW}Converting GMR motions -> CSV: $src -> $out_dir${NC}" >&2
        # Nuke any pre-existing empty/partial cache so we always start clean
        rm -rf "$out_dir"
        mkdir -p "$out_dir"
        if ! python3 "$GMR_CONVERTER_DEFAULT" "$src" "$out_dir" --fps "$GMR_FPS_DEFAULT" >&2; then
            echo -e "${RED}GMR conversion failed${NC}" >&2
            rm -rf "$out_dir"
            return 1
        fi
        # Guard against python exiting 0 without producing any CSVs
        if ! dir_has_csv_motions "$out_dir"; then
            echo -e "${RED}GMR conversion produced no CSV motions in $out_dir${NC}" >&2
            rm -rf "$out_dir"
            return 1
        fi
    fi

    echo "$out_dir"
    return 0
}

if [ ! -e "$MOTION_DATA" ]; then
    echo -e "${RED}Missing motion-data path: $MOTION_DATA${NC}"
    MISSING_FILES=$((MISSING_FILES + 1))
else
    if RESOLVED_MOTION_DATA=$(convert_gmr_if_needed "$MOTION_DATA"); then
        MOTION_DATA="$RESOLVED_MOTION_DATA"
        if [ -d "$MOTION_DATA" ] && dir_has_csv_motions "$MOTION_DATA"; then
            echo -e "${GREEN}Found motion data: $MOTION_DATA${NC}"
        else
            echo -e "${RED}Resolved motion data has no CSV motions: $MOTION_DATA${NC}"
            MISSING_FILES=$((MISSING_FILES + 1))
        fi
    else
        echo -e "${RED}Failed to prepare motion data from: $MOTION_DATA${NC}"
        MISSING_FILES=$((MISSING_FILES + 1))
    fi
fi

if [ $MISSING_FILES -gt 0 ]; then
    echo -e "${YELLOW}Some files are missing. Make sure you have the model files in place.${NC}"
fi

echo ""

# ============================================================================
# Step 2: Install Dependencies (if needed)
# ============================================================================

echo -e "${BLUE}[Step 2/4]${NC} Checking/Installing dependencies..."

if ! command -v just &> /dev/null; then
    echo "Installing dependencies (just not found)..."
    chmod +x scripts/install_deps.sh
    ./scripts/install_deps.sh
else
    echo -e "${GREEN}just is already installed${NC}"
fi

DEPS_OK=true
for cmd in cmake clang git; do
    if ! command -v $cmd &> /dev/null; then
        echo -e "${YELLOW}$cmd not found, will run install_deps.sh${NC}"
        DEPS_OK=false
        break
    fi
done

if [ "$DEPS_OK" = false ]; then
    echo "Installing missing dependencies..."
    chmod +x scripts/install_deps.sh
    ./scripts/install_deps.sh
else
    echo -e "${GREEN}All essential tools are installed${NC}"
fi

echo ""

# ============================================================================
# Step 3: Setup Environment & Build
# ============================================================================

echo -e "${BLUE}[Step 3/4]${NC} Setting up environment and building..."

echo "Sourcing environment setup..."
set +e
source scripts/setup_env.sh
set -e

echo "Building the project..."
just build

echo ""

# ============================================================================
# Step 4: Deploy
# ============================================================================

echo -e "${BLUE}[Step 4/4]${NC} Ready to deploy!"
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}               DEPLOYMENT CONFIGURATION (MoE Student)                  ${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Environment:        ${GREEN}$ENV_TYPE${NC}"
echo -e "  Network Interface:  ${GREEN}$TARGET${NC}"
echo -e "  Policy Model:       ${GREEN}$POLICY${NC}"
echo -e "  Encoder:            ${GREEN}(none - single policy)${NC}"
echo -e "  Motion Data:        ${GREEN}$MOTION_DATA${NC}"
echo -e "  Obs Config:         ${GREEN}$OBS_CONFIG${NC}"
echo -e "  Planner:            ${GREEN}$PLANNER${NC}"
echo -e "  Input Type:         ${GREEN}$INPUT_TYPE${NC}"
echo -e "  Output Type:        ${GREEN}$OUTPUT_TYPE${NC}"
echo -e "  ZMQ Host:           ${GREEN}$ZMQ_HOST${NC}"
if [[ -n "$EXTRA_ARGS" ]]; then
echo -e "  Extra Args:         ${GREEN}$EXTRA_ARGS${NC}"
fi
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}The following command will be executed:${NC}"
echo ""
# No --encoder-file passed: C++ binary skips encoder entirely
echo -e "${BLUE}just run g1_deploy_onnx_ref $TARGET $POLICY $MOTION_DATA \\${NC}"
echo -e "${BLUE}    --obs-config $OBS_CONFIG \\${NC}"
echo -e "${BLUE}    --planner-file $PLANNER \\${NC}"
echo -e "${BLUE}    --input-type $INPUT_TYPE \\${NC}"
echo -e "${BLUE}    --output-type $OUTPUT_TYPE \\${NC}"
echo -e "${BLUE}    --zmq-host $ZMQ_HOST${NC}"
if [[ -n "$EXTRA_ARGS" ]]; then
echo -e "${BLUE}    $EXTRA_ARGS${NC}"
fi
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""

# Ask for confirmation
if [[ "$ENV_TYPE" == "real" ]]; then
    echo -e "${YELLOW}WARNING: This will start the REAL robot control system!${NC}"
else
    echo -e "${YELLOW}This will start the simulation control system.${NC}"
fi
echo ""
read -p "$(echo -e ${GREEN}Proceed with deployment? [Y/n]: ${NC})" confirm

if [[ "$confirm" =~ ^[Yy]$ ]] || [[ -z "$confirm" ]]; then
    echo ""
    echo -e "${GREEN}Starting deployment (MoE Student - single policy)...${NC}"
    echo ""

    # No --encoder-file: encoder is skipped by C++ binary
    if [[ -n "$EXTRA_ARGS" ]]; then
        just run g1_deploy_onnx_ref "$TARGET" "$POLICY" "$MOTION_DATA" \
            --obs-config "$OBS_CONFIG" \
            --planner-file "$PLANNER" \
            --input-type "$INPUT_TYPE" \
            --output-type "$OUTPUT_TYPE" \
            --zmq-host "$ZMQ_HOST" \
            $EXTRA_ARGS
    else
        just run g1_deploy_onnx_ref "$TARGET" "$POLICY" "$MOTION_DATA" \
            --obs-config "$OBS_CONFIG" \
            --planner-file "$PLANNER" \
            --input-type "$INPUT_TYPE" \
            --output-type "$OUTPUT_TYPE" \
            --zmq-host "$ZMQ_HOST"
    fi
else
    echo ""
    echo -e "${YELLOW}Deployment cancelled.${NC}"
    exit 0
fi
