# dump_dots_env.py
import argparse, os, sys, json, re, inspect, importlib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to local DOTS OCR folder")
    ap.add_argument("--out", default="dots_env_report.txt")
    args = ap.parse_args()

    model_dir = os.path.abspath(args.model)
    lines = []

    def add(section, value=""):
        lines.append(f"[{section}] {value}")

    # Basic libs
    try:
        import transformers, torch, huggingface_hub as h
        add("transformers", transformers.__version__)
        add("torch", f"{torch.__version__}  cuda={torch.cuda.is_available()}")
        add("huggingface_hub", h.__version__)
    except Exception as e:
        add("libs_error", repr(e))

    # Model dir contents
    add("model_dir", model_dir)
    if not os.path.isdir(model_dir):
        add("error", "model_dir not found")
    else:
        files = sorted(os.listdir(model_dir))
        add("files", ", ".join(files))

        # Show key JSON processor files if present
        for fname in ["preprocessor_config.json", "image_processor.json", "video_processor.json", "processor_config.json"]:
            p = os.path.join(model_dir, fname)
            if os.path.isfile(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        js = json.load(f)
                    add(f"{fname}", json.dumps(js, ensure_ascii=False)[:1200])
                except Exception as e:
                    add(f"{fname}_error", repr(e))

        # Import config + modeling to discover class names
        try:
            sys.path.insert(0, os.path.dirname(model_dir))
            pkg = os.path.basename(model_dir).replace("-", "_")
            cfg_mod = importlib.import_module(f"{pkg}.configuration_dots")
            mdl_mod = importlib.import_module(f"{pkg}.modeling_dots_ocr")
            add("config_module_file", getattr(cfg_mod, "__file__", "n/a"))
            add("model_module_file", getattr(mdl_mod, "__file__", "n/a"))

            cfg_classes = [n for n,o in cfg_mod.__dict__.items() if isinstance(o, type)]
            mdl_classes = [n for n,o in mdl_mod.__dict__.items() if isinstance(o, type)]
            add("config_classes", ", ".join(cfg_classes))
            add("model_classes", ", ".join(mdl_classes))

            # Likely model class
            likely = [n for n in mdl_classes if re.search(r"(For|Causal|Conditional|LM|Seq|OCR)", n)]
            add("likely_model_classes", ", ".join(likely) if likely else "none")

            # Peek processor wrapper class name from config (if present)
            for k in dir(cfg_mod):
                if k.endswith("Processor"):
                    add("processor_class", k)
        except Exception as e:
            add("import_error", repr(e))

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[ok] wrote {args.out}")

if __name__ == "__main__":
    main()
