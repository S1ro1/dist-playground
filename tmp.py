import torch
from transformers import AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3-8B"


model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
)

model.save_pretrained("out_large")
