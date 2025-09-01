import mss, pyautogui
from PIL import Image, ImageDraw

OUT_RAW = "crosshair_preview_raw.png"
OUT_MARK = "crosshair_preview.png"

def screenshot(path):
    with mss.mss() as sct:
        mon = sct.monitors[1]
        raw = sct.grab(mon)
        img = Image.frombytes("RGB", raw.size, raw.rgb)
        img.save(path, "PNG")
        return img.size

def draw_red_crosshair(src_path, dst_path, x, y, size=40, line_width=3):
    img = Image.open(src_path).convert("RGBA")
    d = ImageDraw.Draw(img)
    # lines
    d.line([(x - size, y), (x + size, y)], fill=(255, 0, 0, 255), width=line_width)
    d.line([(x, y - size), (x, y + size)], fill=(255, 0, 0, 255), width=line_width)
    # X in center
    offset = size // 3
    d.line([(x - offset, y - offset), (x + offset, y + offset)], fill=(255, 0, 0, 255), width=line_width)
    d.line([(x - offset, y + offset), (x + offset, y - offset)], fill=(255, 0, 0, 255), width=line_width)
    img.convert("RGB").save(dst_path, "PNG")
    print(f"[CURSOR] drew red crosshair with X at ({x},{y}) -> {dst_path}")

if __name__ == "__main__":
    w,h = screenshot(OUT_RAW)
    x,y = pyautogui.position()
    print(f"[INFO] Screenshot {w}x{h}, mouse at ({x},{y})")
    draw_red_crosshair(OUT_RAW, OUT_MARK, x, y, size=40)
