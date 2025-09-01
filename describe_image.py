import base64, requests, os

print("[1] Starting script...")

BASE  = "http://127.0.0.1:1234/v1/chat/completions"
MODEL = "google/gemma-3-12b"
IMG   = r"C:\ai-auto\test.png"

print(f"[2] Checking if image exists at: {IMG}")
if not os.path.exists(IMG):
    print("[ERROR] Image file not found! Please put your test.png in C:\\ai-auto")
    exit()

print("[3] Reading image file...")
with open(IMG, "rb") as f:
    img_data = f.read()
print(f"    Image size: {len(img_data)} bytes")

print("[4] Encoding to Base64...")
b64 = base64.b64encode(img_data).decode()
print(f"    Base64 length: {len(b64)}")

print("[5] Building payload...")
payload = {
  "model": MODEL,
  "messages": [
    {"role":"system","content":"You are a concise vision assistant."},
    {"role":"user","content":[
      {"type":"input_text","text":"Describe this screenshot in one sentence."},
      {"type":"input_image","image_url":{"url": f"data:image/png;base64,{b64}"}}
    ]}
  ],
  "temperature": 0.2,
  "max_tokens": 200
}
print("    Payload ready.")

print("[6] Sending POST request to:", BASE)
try:
    r = requests.post(BASE, json=payload, timeout=60)
    print(f"[7] HTTP status: {r.status_code}")
    print(f"[8] Raw response first 500 chars:\n{r.text[:500]}")
    if r.ok:
        try:
            print("[9] Model says:", r.json()["choices"][0]["message"]["content"])
        except Exception as e:
            print("[ERROR] Could not parse model output:", e)
except Exception as e:
    print("[ERROR] Request failed:", e)

print("[10] Script finished.")
