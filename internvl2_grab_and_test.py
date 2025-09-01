# internvl2_grab_and_test.py
# Downloads OpenGVLab/InternVL2-8B into C:\ai-auto\models\InternVL2-8B
# and runs a quick image→text test. Works with GPU (CUDA) or CPU fallback.

import os, sys, argparse, time
from PIL import Image
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# ----- folders you asked for -----
ROOT = r"C:\ai-auto"
MODEL_ID = "OpenGVLab/InternVL2-8B"
LOCAL_DIR = os.path.join(ROOT, "models", "InternVL2-8B")
CACHE_DIR = os.path.join(ROOT, "hf_cache")

# keep HF cache inside C:\ai-auto so you know where it lives
os.environ.setdefault("HF_HOME", CACHE_DIR)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")  # faster download

def log(msg): print(f"[internvl2] {msg}", flush=True)

def pick_device():
    if torch.cuda.is_available():
        d = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        d = "mps"
    else:
        d = "cpu"
    log(f"device = {d}")
    return d

def ensure_model():
    """
    Download the model into LOCAL_DIR if it isn't already there.
    We use the pipeline API first to make sure trust_remote_code bits are handled,
    then save the whole thing to LOCAL_DIR for offline use.
    """
    if os.path.isdir(LOCAL_DIR) and any(os.scandir(LOCAL_DIR)):
        log(f"Found local model: {LOCAL_DIR}")
        return

    log("Downloading model from Hugging Face Hub (first time only)…")
    device = 0 if pick_device() == "cuda" else -1
    # Build the pipeline (this triggers the downloads into HF cache)
    vis_pipe = pipeline(
        "image-text-to-text",
        model=MODEL_ID,
        trust_remote_code=True,
        device=device
    )
    # Persist weights & config to LOCAL_DIR for clean reuse/offline
    log("Saving model & tokenizer to local folder…")
    # Some custom vision models only expose model via pipe.model / pipe.tokenizer
    try:
        vis_pipe.model.save_pretrained(LOCAL_DIR)
        vis_pipe.tokenizer.save_pretrained(LOCAL_DIR)
    except Exception as e:
        log(f"Direct save_pretrained failed ({e}). Trying explicit reload…")
        tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)
        tok.save_pretrained(LOCAL_DIR)
        mdl.save_pretrained(LOCAL_DIR)

    log(f"Saved to {LOCAL_DIR}")

def run_test(image_path: str, question: str):
    """
    Load from LOCAL_DIR (offline friendly) and ask a question about the image.
    """
    device = 0 if pick_device() == "cuda" else -1
    log(f"Loading pipeline from local dir: {LOCAL_DIR}")
    pipe = pipeline(
        "image-text-to-text",
        model=LOCAL_DIR,            # use our local copy
        trust_remote_code=True,
        device=device
    )

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Build the “messages” format expected by InternVL2 in the HF snippet
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": Image.open(image_path).convert("RGB")},
            {"type": "text", "text": question}
        ],
    }]

    log(f"Asking: {question}")
    t0 = time.time()
    out = pipe(text=messages)
    dt = time.time() - t0
    log(f"Answer: {out}")
    log(f"Latency: {dt:.2f}s")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=os.path.join(ROOT, "test.png"),
                    help="Path to an image file to query (default C:\\ai-auto\\test.png)")
    ap.add_argument("--question", default="What do you see? Describe the key buttons.",
                    help="Text question for the image.")
    args = ap.parse_args()

    os.makedirs(LOCAL_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    ensure_model()
    run_test(args.image, args.question)

if __name__ == "__main__":
    main()
