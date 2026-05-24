class ModelNameMap:
    MODEL_NAME_MAP = {
        "llama3.1": "meta-llama/Llama-3.1-8B-Instruct",
        "llama3.3": "meta-llama/Llama-3.3-70B-Instruct",
        "qwen3" : "Qwen/Qwen3-4B-Instruct-2507",
        "qwen3-next": "Qwen/Qwen3-Next-80B-A3B-Instruct"
    }

    @classmethod
    def get_model_name(cls, key: str) -> str:
        return cls.MODEL_NAME_MAP.get(key)