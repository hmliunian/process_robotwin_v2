python := ".venv/bin/python"
qwen_python := "../process_data/.venv-qwen35/bin/python"
qwen_model := "checkpoints/Qwen/Qwen3.5-27B"
qwen_min_free_mib := "60000"
qwen_startup_timeout := "600"
config := "configs/pilot_move_pillbottle_pad.yaml"
process_config := "configs/process.yaml"

set positional-arguments := true

default:
    @just --list

test:
    {{python}} -m pytest tests/unit -q

test-all:
    {{python}} -m pytest -q

lint:
    {{python}} -m ruff check src tests scripts

format:
    {{python}} -m ruff format src tests scripts

preflight:
    {{python}} scripts/run_target_receiver.py preflight --config {{config}}

loop episode_id="7152":
    {{python}} scripts/run_target_receiver.py loop --config {{config}} --episode {{episode_id}}

serve-qwen:
    env PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" {{python}} scripts/manage_qwen_process.py --serve-only --config {{config}} --qwen-python {{qwen_python}} --qwen-model-path {{qwen_model}} --qwen-min-free-memory-mib {{qwen_min_free_mib}} --qwen-startup-timeout {{qwen_startup_timeout}}

qwen episode_id="7152":
    {{python}} scripts/run_target_receiver.py qwen --config {{config}} --episode {{episode_id}}

sam run_id episode_id="7152":
    {{python}} scripts/run_target_receiver.py sam --config {{config}} --episode {{episode_id}} --run-id {{run_id}}

run episode_id="7152":
    {{python}} scripts/run_target_receiver.py run --config {{config}} --episode {{episode_id}}

process *process_args:
    @dataset_path=""; output_dir=""; if [ "$#" -gt 0 ] && [ "${1#-}" = "$1" ]; then dataset_path="$1"; shift; fi; if [ "$#" -gt 0 ] && [ "${1#-}" = "$1" ]; then output_dir="$1"; shift; fi; if [ -n "$output_dir" ]; then set -- --output-dir "$output_dir" "$@"; fi; if [ -n "$dataset_path" ]; then set -- --data-path "$dataset_path" "$@"; fi; exec env PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" {{quote(python)}} scripts/manage_qwen_process.py --config {{quote(process_config)}} --qwen-python {{quote(qwen_python)}} --qwen-model-path {{quote(qwen_model)}} --qwen-min-free-memory-mib {{quote(qwen_min_free_mib)}} --qwen-startup-timeout {{quote(qwen_startup_timeout)}} -- "$@"

# Run Qwen API requests with one persistent SAM worker per configured GPU.
run-parallel *process_args:
    @if [ -z "${SAM_WORKER_GPUS:-}" ]; then echo "SAM_WORKER_GPUS is required (for example: 0,1,2,3)" >&2; exit 2; fi; exec just process "$@" --sam-worker-gpus "$SAM_WORKER_GPUS" --qwen-max-in-flight "${QWEN_MAX_IN_FLIGHT:-4}"

# Convert reviewed real pick-and-place MCAPs to the RoboTwin task layout.
convert-real input_root output_root *convert_args:
    @exec env PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" {{quote(python)}} scripts/convert_real_mcap_dataset.py "$@"

check-gpu:
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
