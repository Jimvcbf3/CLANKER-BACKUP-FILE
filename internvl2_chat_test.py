# internvl2_chat_test.py
# InternVL2-8B local chat: manual/auto image preprocessing → model.chat() with dict gen_config

import torch
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from PIL import Image
import numpy as np

MODEL_DIR  = r"C:\ai-auto\InternVL2-8B"
IMAGE_PATH = r"C:\ai-auto\test.png"
QUESTION   = "Which button should I click to open Groups? Answer briefly."

# Standard CLIP-like normalization used by many VLMs
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def to_pixel_values(img: Image.Image, size: int) -> torch.Tensor:
    # resize: shorter side -> size, keep aspect, then center-crop to size x size
    w, h = img.size
    if w < h:
        new_w = size
        new_h = int(h * (size / w))
    else:
        new_h = size
        new_w = int(w * (size / h))
    img_resized = img.resize((new_w, new_h), Image.BICUBIC)

    left = (new_w - size) // 2
    top  = (new_h - size) // 2
    img_cropped = img_resized.crop((left, top, left + size, top + size))

    arr = np.asarray(img_cropped).astype("float32") / 255.0
    arr = (arr - np.array(MEAN)) / np.array(STD)
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    tensor = torch.from_numpy(arr)       # [3, size, size]
    tensor = tensor.unsqueeze(0)         # [1, 3, size, size]
    return tensor

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32
    print(f"[boot] device={device} dtype={dtype}")

    print("[load] tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    print("[load] model…")
    model = AutoModel.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        torch_dtype=dtype
    ).eval()
    if device != "cuda":
        model.to(device)

    # Determine target image_size (fallback 448)
    image_size = None
    if hasattr(model, "vision_tower") and hasattr(model.vision_tower, "config"):
        image_size = getattr(model.vision_tower.config, "image_size", None)
    if image_size is None and hasattr(model, "config"):
        for key in ["vision_config", "vision", "image_size"]:
            cfg = getattr(model.config, key, None)
            if isinstance(cfg, int):
                image_size = cfg
                break
            if hasattr(cfg, "image_size"):
                image_size = getattr(cfg, "image_size")
                break
    if image_size is None:
        image_size = 448
    print(f"[img] target image_size = {image_size}")

    # Try AutoImageProcessor; else manual preprocessing
    pixel_values = None
    try:
        print("[proc] trying AutoImageProcessor…")
        img_proc = AutoImageProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
        out = img_proc(images=Image.open(IMAGE_PATH).convert("RGB"), return_tensors="pt")
        if "pixel_values" in out:
            pixel_values = out["pixel_values"]
            print(f"[proc] AutoImageProcessor ok. shape={tuple(pixel_values.shape)}")
    except Exception as e:
        print("[proc] AutoImageProcessor not usable:", repr(e))

    if pixel_values is None:
        print("[proc] falling back to manual preprocessing…")
        img = Image.open(IMAGE_PATH).convert("RGB")
        pixel_values = to_pixel_values(img, size=image_size)
        print(f"[proc] manual pixel_values shape={tuple(pixel_values.shape)}")

    pixel_values = pixel_values.to(device=device, dtype=dtype)

    # ✅ Use a dict for generation_config (NOT a GenerationConfig object)
    gen_cfg = {
        "max_new_tokens": 128,
        # you can also add: "temperature": 0.2, "top_p": 0.9, etc.
    }

    print("[chat] model.chat(tokenizer, pixel_values, QUESTION, gen_cfg)")
    out = model.chat(
        tokenizer,
        pixel_values,
        QUESTION,
        gen_cfg,
        history=None,
        return_history=False,
        verbose=False
    )

    print("\n[answer]")
    print(out)

if __name__ == "__main__":
    main()
