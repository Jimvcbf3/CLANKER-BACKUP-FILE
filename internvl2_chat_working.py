# internvl2_chat_working.py
# InternVL2-8B fully local, uses the model's custom chat() with an image.
# If test.png is missing, we create a tiny dummy image so it won't assert.

import os, time
from PIL import Image, ImageDraw
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR  = r"C:\ai-auto\InternVL2-8B"
TEST_IMAGE = r"C:\ai-auto\test.png"   # put a real screenshot here if you have one
QUESTION   = "Which button on this screen opens Groups? Answer briefly."

def ensure_image(path):
    if os.path.isfile(path):
        return Image.open(path).convert("RGB")
    # make a 64x64 dummy with a red X so the model definitely gets an image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.new("RGB", (64, 64), (40, 40, 40))
    d = ImageDraw.Draw(img)
    d.line((0,0,63,63), fill=(220,60,60), width=5)
    d.line((0,63,63,0), fill=(220,60,60), width=5)
    img.save(path)
    return img

def pick_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def main():
    device = pick_device()
    dtype  = torch.float16 if device == "cuda" else torch.float32
    print(f"[boot] device={device} dtype={dtype}")

    print("[load] tokenizer…")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    print("[load] model… (first time can be slow)")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None
    )
    if device != "cuda":
        model.to(device)
    model.eval()

    # Always provide an image to avoid the img_context assertion
    img = ensure_image(TEST_IMAGE)

    print("[infer] calling model.chat(query, image)…")
    t0 = time.time()
    # InternVL2 exposes a custom chat() when trust_remote_code=True
    out = model.chat(
        tok,
        query=QUESTION,
        image=img,
        max_new_tokens=128
    )
    dt = time.time() - t0
    print("\n[answer]", out)
    print(f"[latency] {dt:.2f}s")

if __name__ == "__main__":
    main()
