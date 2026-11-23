"""Produce artifact.png — the image you keep.

Plumbing, written for you. It calls your own code, so it only works once your
app does, and it is deliberately defensive: an unfinished level makes it print
what is missing rather than raise, so a clean import stays possible while you
are still building.

    python make_artifact.py

The grader runs this too, which is what stops artifact.png from being any
image you happened to have lying around.
"""
from __future__ import annotations

from pathlib import Path

PROVIDED = Path(__file__).resolve().parent / "provided"
OUT = Path(__file__).resolve().parent / "artifact.png"
MODE = "overlay"       # "blur", "count" or "overlay" -- your call


def build(out_path=OUT, mode=MODE):
    """Run the pipeline over one image and save the annotated result."""
    import csv
    import numpy as np
    from PIL import Image

    from facedetect.faces import load_face_detector, detect_faces
    from facedetect.app import process

    rows = [r for r in csv.DictReader(open(PROVIDED / "faces.csv"))
            if r["group"] == "near"]
    src = PROVIDED / "faces" / rows[0]["filename"]

    det = load_face_detector()
    try:
        found = detect_faces(det, src)
    finally:
        try:
            det.close()
        except Exception:
            pass

    img = np.asarray(Image.open(src).convert("RGB"))
    annotated, report = process(img, found, mode)
    Image.fromarray(np.asarray(annotated, dtype=np.uint8)).save(out_path)
    return report


def main() -> int:
    try:
        report = build()
    except NotImplementedError as e:
        print(f"not ready yet: {e}")
        return 1
    print(f"wrote {OUT.name}  {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
