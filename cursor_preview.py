# cursor_preview.py
# pip install mss pillow pyautogui
import os, time, base64
import mss, pyautogui
from PIL import Image, ImageDraw

OUT_RAW = "cursor_preview_raw.png"
OUT_PNG = "cursor_preview.png"

def screenshot(path):
    with mss.mss() as sct:
        mon = sct.monitors[1]
        raw = sct.grab(mon)
        img = Image.frombytes("RGB", raw.size, raw.rgb)
        img.save(path, "PNG")
        return img.size

def draw_red_cursor_arrow(src_path, dst_path, x, y, size=36, outline=3):
    img = Image.open(src_path).convert("RGBA")
    d = ImageDraw.Draw(img)
    shaft_len = int(size * 1.8); shaft_w = max(4, size // 6)
    head_w = int(size * 0.9); head_len = int(size * 0.9)
    dx, dy = -1, -1  # point up-left
    px, py = dy, -dx
    half = shaft_w // 2
    x2 = x + dx * shaft_len; y2 = y + dy * shaft_len
    p1 = (x + px*half, y + py*half); p2 = (x - px*half, y - py*half)
    p3 = (x2 - px*half, y2 - py*half); p4 = (x2 + px*half, y2 + py*half)
    hx = x + dx * head_len; hy = y + dy * head_len
    h1=(x,y); h2=(hx + py * head_w//2, hy - px * head_w//2); h3=(hx - py * head_w//2, hy + px * head_w//2)
    if outline>0:
        d.polygon([p1,p4,p3,p2], fill=None, outline=(255,255,255,255), width=outline+1)
        d.polygon([h1,h2,h3], fill=None, outline=(255,255,255,255), width=outline+1)
    red=(255,0,0,255)
    d.polygon([p1,p4,p3,p2], fill=red, outline=red, width=1)
    d.polygon([h1,h2,h3], fill=red, outline=red, width=1)
    img.convert("RGB").save(dst_path, "PNG")

if __name__ == "__main__":
    w,h = screenshot(OUT_RAW)
    x,y = pyautogui.position()
    print(f"[INFO] screen {w}x{h}, current mouse at ({x},{y})")
    draw_red_cursor_arrow(OUT_RAW, OUT_PNG, x, y, size=38, outline=3)
    print(f"[DONE] wrote {OUT_PNG}")
