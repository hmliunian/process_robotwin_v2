python := ".venv/bin/python"
qwen_python := "../process_data/.venv-qwen35/bin/python"
config := "configs/pilot_move_pillbottle_pad.yaml"

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

check-gpu:
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
