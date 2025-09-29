#!/usr/bin/env python3
"""
Lightweight model downloader/initializer.

- Creates model directories
- Optionally downloads YOLO and GridFormer weights from given URLs
- Prints clear instructions if URLs are not provided

Usage examples:
  python scripts/download_models.py \
    --yolo-url https://example.com/yolov8s.onnx \
    --gridformer-url https://example.com/gridformer.onnx

Notes:
- If you prefer Ultralytics auto-download, run inside Python:
    from ultralytics import YOLO
    YOLO("yolov8s.pt")
  and then move the weight into models/yolo/
"""

from __future__ import annotations
import argparse
import os
import sys
import urllib.request
from pathlib import Path
from typing import Optional


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download(url: str, out_path: Path) -> None:
    def _progress(block_num: int, block_size: int, total_size: int):
        downloaded = block_num * block_size
        percent = 0 if total_size == 0 else min(100, downloaded * 100 // total_size)
        print(f"\rDownloading {out_path.name}: {percent}%", end="", flush=True)

    print(f"\n➡️  Downloading from: {url}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, out_path.as_posix(), _progress)
    print(f"\n✅ Saved to: {out_path}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download or prepare model folders")
    parser.add_argument("--output-dir", default="models", help="Base models directory")
    parser.add_argument("--yolo-url", default=None, help="Direct URL for YOLO weight (pt/onnx)")
    parser.add_argument("--gridformer-url", default=None, help="Direct URL for GridFormer (onnx/pth)")
    parser.add_argument("--force", action="store_true", help="Overwrite if file exists")
    args = parser.parse_args(argv)

    base_dir = Path(args.output_dir)
    yolo_dir = base_dir / "yolo"
    gridformer_dir = base_dir / "gridformer"

    ensure_dir(yolo_dir)
    ensure_dir(gridformer_dir)

    print("\n📂 Prepared model directories:")
    print(f" - {yolo_dir}")
    print(f" - {gridformer_dir}")

    # YOLO
    if args.yolo_url:
        yolo_name = Path(args.yolo_url).name
        yolo_out = yolo_dir / yolo_name
        if yolo_out.exists() and not args.force:
            print(f"⚠️  YOLO weight exists, skipping: {yolo_out} (use --force to overwrite)")
        else:
            download(args.yolo_url, yolo_out)
    else:
        print("\nℹ️  YOLO weight URL not provided. Options:")
        print(" - Provide --yolo-url <direct_download_url>")
        print(" - Or auto-download with Ultralytics and move file into models/yolo/")
        print("     from ultralytics import YOLO; YOLO('yolov8s.pt')")

    # GridFormer
    if args.gridformer_url:
        gf_name = Path(args.gridformer_url).name
        gf_out = gridformer_dir / gf_name
        if gf_out.exists() and not args.force:
            print(f"⚠️  GridFormer weight exists, skipping: {gf_out} (use --force to overwrite)")
        else:
            download(args.gridformer_url, gf_out)
    else:
        print("\nℹ️  GridFormer URL not provided. Place your ONNX/PyTorch file into:")
        print(f" - {gridformer_dir}")

    print("\n✅ Done. Current contents:")
    for root, _, files in os.walk(base_dir):
        for f in files:
            print(f" - {Path(root) / f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
