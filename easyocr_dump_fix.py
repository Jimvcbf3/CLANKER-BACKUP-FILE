# easyocr_dump_fix.py
# Same as before, but cast results to JSON-safe Python types.

import sys, os, json
import easyocr

LANGS = ['en', 'ch_tra']

def py_bbox(bbox):
    # bbox is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] possibly as numpy
    return [[int(p[0]), int(p[1])] for p in bbox]

def main():
    if len(sys.argv) < 2:
        print("Usage: py -u easyocr_dump_fix.py <image_path> [--gpu]")
        sys.exit(1)

    img_path = sys.argv[1]
    use_gpu = ('--gpu' in sys.argv)

    if not os.path.exists(img_path):
        print(f"[error] image not found: {img_path}")
        sys.exit(1)

    print(f"[init] EasyOCR Reader langs={LANGS} gpu={use_gpu}")
    reader = easyocr.Reader(LANGS, gpu=use_gpu)

    print(f"[run] reading: {img_path}")
    results = reader.readtext(img_path, detail=1)  # [ [bbox, text, conf], ... ]

    print(f"[ok] {len(results)} items")
    out = []
    for i, item in enumerate(results, 1):
        bbox, text, conf = item
        out.append({
            "idx": int(i),
            "bbox": py_bbox(bbox),
            "text": str(text),
            "confidence": float(conf)
        })

    # pretty print JSON (now serializable)
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
