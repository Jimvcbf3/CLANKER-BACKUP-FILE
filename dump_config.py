# dump_config.py — safe inspection only
# Usage:
#   py -u dump_config.py --model C:\ai-auto\dots_ocr

import os, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    return ap.parse_args()

def resolve_image_token_id(tokenizer, model):
    # 1) numeric in config
    for key in ("image_token_index", "image_token_id"):
        val = getattr(model.config, key, None)
        if isinstance(val, int) and val >= 0:
            return key, val

    # 2) string placeholders that already exist in vocab
    candidates = []
    for key in ("image_token", "visual_token"):
        v = getattr(model.config, key, None)
        if isinstance(v, str) and v:
            candidates.append(v)
    candidates += ["<image>", "<img>", "<image_token>"]

    for s in candidates:
        tid = tokenizer.convert_tokens_to_ids(s)
        if tid is not None and tid >= 0 and tid != tokenizer.unk_token_id:
            return f"vocab:{s}", tid

    # 3) not found
    return "not-found", None

def main():
    args = parse_args()
    model_dir = os.path.abspath(args.model)
    print(f"[info] Loading model + tokenizer from {model_dir} ...")
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=True)

    print("\n===== MODEL CONFIG (keys of interest) =====")
    for k in [
        "model_type", "vision_tower", "image_token", "visual_token",
        "image_token_index", "image_token_id", "spatial_merge_size"
    ]:
        print(f"{k}: {getattr(model.config, k, None)}")

    print("\n===== TOKENIZER SPECIAL TOKENS =====")
    print("special_tokens_map:", tok.special_tokens_map)
    print("additional_special_tokens:", getattr(tok, "additional_special_tokens", None))

    print("\n===== CHAT TEMPLATE SUPPORT =====")
    has_apply = hasattr(tok, "apply_chat_template")
    print("tokenizer.apply_chat_template:", has_apply)
    if has_apply:
        tmpl = getattr(tok, "chat_template", None)
        print("chat_template present:", bool(tmpl))
        if tmpl:
            print("chat_template (first 400 chars):")
            print(str(tmpl)[:400])

    print("\n===== PLACEHOLDER PROBES =====")
    probes = ["<image>", "<img>", "<image_token>",
              getattr(model.config, "image_token", None),
              getattr(model.config, "visual_token", None)]
    seen = set()
    for p in probes:
        if not p or p in seen:
            continue
        seen.add(p)
        tid = tok.convert_tokens_to_ids(p)
        print(f"{p!r} -> token_id: {tid}")

    src, resolved = resolve_image_token_id(tok, model)
    print("\n===== RESOLVED IMAGE TOKEN ID =====")
    print(f"source: {src}  |  token_id: {resolved}")

if __name__ == "__main__":
    main()