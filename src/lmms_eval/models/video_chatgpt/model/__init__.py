try:
    from lmms_eval.models.video_chatgpt.model.video_chatgpt import VideoChatGPTLlamaForCausalLM, VideoChatGPTConfig
except ImportError as e:
    print(f"An error occurred while importing modules: {e}")