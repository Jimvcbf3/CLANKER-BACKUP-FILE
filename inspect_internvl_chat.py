import inspect
from transformers import AutoTokenizer, AutoModel
import torch

MODEL_DIR = r"C:\ai-auto\InternVL2-8B"

print("[*] Loading tokenizer…")
tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

print("[*] Loading model… (can be slow)")
model = AutoModel.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    torch_dtype=torch.float32
)

print("\n[+] Inspecting model.chat signature…")
print(inspect.signature(model.chat))

doc = inspect.getdoc(model.chat)
print("\n--- chat.__doc__ ---")
print(doc if doc else "No docstring found")
