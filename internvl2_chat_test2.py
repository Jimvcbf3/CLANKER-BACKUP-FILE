import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16  # or torch.bfloat16 if your GPU supports it; 4060 does FP16 fine.

model = InternVLChatModel.from_pretrained(
    LOCAL_DIR,
    trust_remote_code=True
).to(device=device, dtype=dtype)
