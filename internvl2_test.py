# internvl2_test.py
# Minimal local run for InternVL2-8B (text + multimodal)

from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import torch

def main():
    model_id = "C:/ai-auto/InternVL2-8B"  # path to your downloaded model

    print("[*] Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    print("[*] Loading model… (this can take minutes)")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,   # or bfloat16 if your GPU supports it
        device_map="auto",
        trust_remote_code=True
    )

    # === Example 1: text only ===
    messages = [
        {"role": "user", "content": "Tell me a short funny fact about computers."}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    output = model.generate(**inputs, max_new_tokens=64)
    print("\n[TEXT ONLY OUTPUT]")
    print(tokenizer.decode(output[0], skip_special_tokens=True))

    # === Example 2: multimodal (image + text) ===
    try:
        image = Image.open("test.png")   # replace with your screenshot file
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What button should I click to open Groups?"}
            ]}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        output = model.generate(**inputs, max_new_tokens=128)
        print("\n[MULTIMODAL OUTPUT]")
        print(tokenizer.decode(output[0], skip_special_tokens=True))
    except Exception as e:
        print(f"[WARN] Skipping image test: {e}")

if __name__ == "__main__":
    main()
