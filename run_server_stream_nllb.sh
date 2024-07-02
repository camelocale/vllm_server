python -m vllm.entrypoints.openai.api_server \
    --model /home/user/.cache/huggingface/hub/models--Unbabel--TowerInstruct-7B-v0.2/snapshots/6ea1d9e6cb5e8badde77ef33d61346a31e6ff1d4 \
    --dtype auto --api-key sk-MAL7TDoiDS09yLjOnEkcT3BlbkFJfHrtwfndr8XP7NYkk2Yvls \
    --chat-template /home/user/code/vllm/examples/template_chatml.jinja \
    --port 8080


## --model /home/user/.cache/huggingface/hub/models--Unbabel--TowerInstruct-7B-v0.2/snapshots/6ea1d9e6cb5e8badde77ef33d61346a31e6ff1d4 \
## --model /home/user/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/a8977699a3d0820e80129fb3c93c20fbd9972c41 \
