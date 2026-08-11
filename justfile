python := ".venv/bin/python"
qwen_python := "../process_data/.venv-qwen35/bin/python"
config := "configs/pilot_move_pillbottle_pad.yaml"

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
    {{qwen_python}} scripts/serve_qwen.py --model checkpoints/Qwen/Qwen3.5-27B --device cuda:0

qwen episode_id="7152":
    {{python}} scripts/run_target_receiver.py qwen --config {{config}} --episode {{episode_id}}

sam run_id episode_id="7152":
    {{python}} scripts/run_target_receiver.py sam --config {{config}} --episode {{episode_id}} --run-id {{run_id}}

run episode_id="7152":
    {{python}} scripts/run_target_receiver.py run --config {{config}} --episode {{episode_id}}

process dataset_root="../dataset/move_pillbottle_pad_coverage20_original" output_dir="artifacts/runs" *process_args:
    @dataset_root="$1"; output_dir="$2"; shift 2; if [ "${output_dir#-}" != "$output_dir" ]; then set -- "$output_dir" "$@"; output_dir="artifacts/runs"; fi; exec env PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" {{quote(python)}} scripts/process_dataset.py --config {{quote(config)}} --dataset-root "$dataset_root" --output-dir "$output_dir" "$@"

check-gpu:
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
