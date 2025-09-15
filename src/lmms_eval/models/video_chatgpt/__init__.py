try:
    from .model import VideoChatGPTLlamaForCausalLM
except ImportError as e:
    print(f"An error occurred while importing modules: {e}")