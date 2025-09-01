# easyocr_annotate.py
# pip install easyocr supervision opencv-python numpy
# Optional (for GPU speed): torch with CUDA wheels installed

import argparse, os, json, sys
import numpy as np
import cv2
import easyocr
import supervision as sv

def parse_args():
    p = argparse.ArgumentParser("EasyOCR + annotate + XY print")
    p.add_argument("--image", required=True, help="Path to image (PNG/JPG).")
    p.add_argument("--langs", default="en,ch_tra",
                   help="Comma-separated langs, e.g. en,ch_tra or en,zh_cn")
    p.add_argument("--min-conf", type=float, default=0.30,
                   help="Minimum confidence to keep a detection.")
    p.add_argument("--gpu", action="store_true", help="Use GPU if available (requires CUDA torch).")
    p.add_argument("--no-show", action="store_true", help="Do not open a preview window.")
    p.add_argument("--out-prefix", default=None,
                   help="Prefix for outputs (default = <image_stem>)")
    return p.parse_args()

def main():
    args = parse_args()
    img_path = args.image
    if not os.path.isfile(img_path):
        print(f"[err] image not found: {img_path}")
        sys.exit(1)

    langs = [s.strip() for s in args.langs.split(",") if s.strip()]
    stem = args.out_prefix or os.path.splitext(os.path.basename(img_path))[0]
    out_img = f"{stem}_annotated.jpg"
    out_json = f"{stem}_ocr.json"

    print(f"[init] EasyOCR Reader langs={langs} gpu={args.gpu}")
    reader = easyocr.Reader(langs, gpu=args.gpu)

    print(f"[run] reading: {img_path}")
    results = reader.readtext(img_path)   # list of (bbox, text, conf)

    print(f"[ok] detections: {len(results)} (pre-filter)")

    # Build detections for visualization
    xyxy, confs, cls_ids, labels = [], [], [], []
    dump_rows = []
    kept = 0

    for i, det in enumerate(results, start=1):
        bbox, text, conf = det  # bbox = 4 corner points [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        xs = [float(p[0]) for p in bbox]
        ys = [float(p[1]) for p in bbox]
        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))
        conf_f = float(conf)

        if conf_f < args.min_conf:
            continue

        kept += 1
        # center
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        xyxy.append([x1, y1, x2, y2])
        confs.append(conf_f)
        cls_ids.append(0)
        labels.append(f"{text} ({conf_f:.2f})")

        # Print to CMD for your automation use
        print(f"[det {kept}] text='{text}'  conf={conf_f:.2f}  xyxy=({x1},{y1},{x2},{y2})  center=({cx},{cy})")

        dump_rows.append({
            "idx": int(kept),
            "bbox": [[int(p[0]), int(p[1])] for p in bbox],
            "xyxy": [int(x1), int(y1), int(x2), int(y2)],
            "center": [int(cx), int(cy)],
            "text": text,
            "confidence": conf_f
        })

    print(f"[filter] kept: {kept} (min_conf={args.min_conf:.2f})")

    # Save JSON
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "image": os.path.basename(img_path),
            "langs": langs,
            "min_conf": args.min_conf,
            "kept": kept,
            "detections": dump_rows
        }, f, ensure_ascii=False, indent=2)
    print(f"[save] {out_json}")

    # Draw boxes + labels
    image = cv2.imread(img_path)
    if image is None:
        print("[err] cv2 failed to read the image for drawing.")
        sys.exit(1)

    dets = sv.Detections(
        xyxy=np.array(xyxy, dtype=np.int32) if xyxy else np.zeros((0, 4), dtype=np.int32),
        confidence=np.array(confs, dtype=np.float32) if confs else np.zeros((0,), dtype=np.float32),
        class_id=np.array(cls_ids, dtype=np.int32) if cls_ids else np.zeros((0,), dtype=np.int32)
    )

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated = box_annotator.annotate(scene=image.copy(), detections=dets)
    annotated = label_annotator.annotate(scene=annotated, detections=dets, labels=labels)

    cv2.imwrite(out_img, annotated)
    print(f"[save] {out_img}")

    if not args.no_show:
        cv2.imshow("EasyOCR Annotated", annotated)
        print("[show] press any key to close window…")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
