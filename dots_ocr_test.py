# dots_ocr_test.py
# Local runner for rednote-hilab/dots.ocr with dynamic class discovery.
# Use:
#   cd /d C:\ai-auto
#   py -u dots_ocr_test.py --image C:/ai-auto/test.png --model C:/ai-auto/dots_ocr

import argparse, os, sys, json
from importlib import import_module
from PIL import Image
import torch

CANDIDATE_CLASS_NAMES = [
    "DotsOCRForConditionalGeneration",
    "DotsOCRForImageTextToText",
    "DotsForConditionalGeneration",
    "DotsOCRModel",
    "DotsOCR",
]

def make_package(model_path: str):
    pkg_name = os.path.basename(model_path.rstrip("\\/"))
    parent = os.path.dirname(model_path)
    init_py = os.path.join(model_path, "__init__.py")
    if not os.path.exists(init_py):
        try:
            with open(init_py, "w", encoding="utf-8") as f:
                f.write("# package marker for DOTS OCR\n")
        except Exception as e:
            print(f"[WARN] could not create __init__.py: {e}")
    if parent not in sys.path:
        sys.path.insert(0, parent)
    return pkg_name

def pick_model_class(mdl_mod):
    # 1) try known names
    for name in CANDIDATE_CLASS_NAMES:
        if hasattr(mdl_mod, name):
            cls = getattr(mdl_mod, name)
            if hasattr(cls, "from_pretrained"):
                print(f"[pick] using class by name: {name}")
                return cls
    # 2) search for any class with from_pretrained + generate
    for name in dir(mdl_mod):
        obj = getattr(mdl_mod, name)
        if isinstance(obj, type) and hasattr(obj, "from_pretrained"):
            # bonus if it looks generative
            if hasattr(obj, "generate"):
                print(f"[pick] discovered generative class: {name}")
                return obj
    # 3) last resort: any class with from_pretrained
    for name in dir(mdl_mod):
        obj = getattr(mdl_mod, name)
        if isinstance(obj, type) and hasattr(obj, "from_pretrained"):
            print(f"[pick] fallback class: {name}")
            return obj
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", required=True, help="folder with DOTS files & weights")
    ap.add_argument("--max-new", type=int, default=256)
    args = ap.parse_args()

    model_dir = os.path.abspath(args.model)
    img_path  = os.path.abspath(args.image)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32
    print(f"[boot] device={device} dtype={dtype}")

    if not os.path.isdir(model_dir):
        print(f"[ERR ] Model folder not found: {model_dir}")
        return

    needed = [
        "configuration_dots.py",
        "modeling_dots_ocr.py",
        "preprocessor_config.json",
        "model.safetensors.index.json",
    ]
    missing = [f for f in needed if not os.path.exists(os.path.join(model_dir, f))]
    if missing:
        print(f"[ERR ] Missing files: {', '.join(missing)}")
        return

    # package-ize and import
    pkg = make_package(model_dir)
    try:
        cfg_mod = import_module(f"{pkg}.configuration_dots")
        mdl_mod = import_module(f"{pkg}.modeling_dots_ocr")
    except Exception as e:
        print(f"[ERR ] import failed: {e}")
        print(f"[hint] Folder must be named '{pkg}' and contain the DOTS files.")
        return

    # dynamic class pick
    ModelClass = pick_model_class(mdl_mod)
    if ModelClass is None:
        print("[ERR ] Could not find a model class with from_pretrained().")
        print("[info] Available names in module:")
        print(", ".join(sorted([n for n in dir(mdl_mod) if not n.startswith('_')])))
        return

    # load processor/tokenizer
    from transformers import AutoProcessor, AutoTokenizer
    print(f"[load] processor from {model_dir}")
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

    print(f"[load] model class={ModelClass.__name__}")
    model = ModelClass.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()

    # image → inputs
    img = Image.open(img_path).convert("RGB")

    # Try several input signatures (some processors want 'text', some don't)
    tries = [
        dict(images=img, return_tensors="pt"),
        dict(images=img, text="Read all visible text.", return_tensors="pt"),
        dict(images=img, text="OCR this image and output the text only.", return_tensors="pt"),
    ]
    inputs = None
    last_err = None
    for k in tries:
        try:
            tmp = processor(**k)
            if isinstance(tmp, dict) and len(tmp):
                inputs = tmp
                break
        except Exception as e:
            last_err = e
    if inputs is None:
        print("[ERR ] processor() failed to produce inputs.")
        if last_err:
            print(f"[hint] last error: {repr(last_err)}")
        return

    # push to device
    for k, v in list(inputs.items()):
        if hasattr(v, "to"):
            inputs[k] = v.to(device)

    # generate with a few shapes
    out_ids = None
    gen_errors = []
    for gkw in [
        dict(max_new_tokens=args.max_new, do_sample=False),
        dict(text="Read text.", max_new_tokens=args.max_new, do_sample=False),
    ]:
        try:
            with torch.inference_mode():
                out_ids = model.generate(**inputs, **gkw)
            break
        except Exception as e:
            gen_errors.append(repr(e))

    if out_ids is None:
        print("[ERR ] model.generate() failed with all attempts.")
        for i, err in enumerate(gen_errors, 1):
            print(f"[gen err {i}] {err}")
        return

    # decode
    if hasattr(processor, "tokenizer") and processor.tokenizer:
        text = processor.tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]
    else:
        tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        text = tok.batch_decode(out_ids, skip_special_tokens=True)[0]

    print("\n[ocr] -------- BEGIN TEXT --------")
    print(text.strip())
    print("[ocr] --------- END TEXT ---------")

    # Optional: try to parse JSON if it emits structured output
    try:
        data = json.loads(text)
        print(f"\n[json] parsed: {type(data).__name__}")
    except Exception:
        pass

if __name__ == "__main__":
    main()
