# internvl2_local_chat.py
# Load InternVL2-8B from a local folder and use its custom chat() API.
# Works on CPU; will try CUDA if available.

import os, time
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_DIR = r"C:\ai-auto\InternVL2-8B"   # your downloaded folder
TEST_IMAGE = r"C:\ai-auto\test.png"      # put any screenshot/image here
QUESTION   = "Describe the key buttons. Which one opens Groups?"

def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def main():
    device = pick_device()
    print(f"[boot] device = {device}")

    # dtype: use float16 on cuda, float32 on cpu
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("[load] tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, trust_remote_code=True
    )

    print("[load] model… (first time can be slow)")
    model = AutoModel.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        torch_dtype=dtype
    )
    model.to(device)
    model.eval()

    # prepare message with image
    if not os.path.isfile(TEST_IMAGE):
        raise FileNotFoundError(f"TEST_IMAGE not found: {TEST_IMAGE}")
    image = Image.open(TEST_IMAGE).convert("RGB")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": QUESTION}
        ]
    }]

    print("[infer] running chat() …")
    t0 = time.time()
    # InternVL2 exposes a custom chat() when trust_remote_code=True
    out = model.chat(tokenizer, messages=messages, max_new_tokens=128)
    dt = time.time() - t0
    print("\n[answer]", out)
    print(f"[latency] {dt:.2f}s")

if __name__ == "__main__":
    main()
