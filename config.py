from pathlib import Path

VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8000
VLLM_API_KEY = "token-abc123"

HF_HOME = str(Path.home() / ".cache" / "huggingface")

AVAILABLE_MODELS = {
    "Qwen2.5 1.5B Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen2.5 3B Instruct": "Qwen/Qwen2.5-3B-Instruct",
    "Qwen2.5 7B Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "Qwen3-30B-A3B-Instruct-2507-FP8": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
    "Ministral 3 14B Instruct 2512 BF16": "mistralai/Ministral-3-14B-Instruct-2512-BF16",

}

DEFAULT_MAX_MODEL_LEN = 2048
DEFAULT_GPU_MEMORY_UTILIZATION = 0.70
DEFAULT_DTYPE = "float16"
