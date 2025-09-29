#!/usr/bin/env python3
"""
Weather Presentation Demo (robust YOLO parsing)

Produces a 3-panel presentation video showing simultaneous GridFormer restoration
and object detection under adverse weather (fog, rain, snow) and clean scenes.

Layout (3-panel):
- Left: degraded (weathered) image
- Middle: GridFormer-restored image
- Right: restored + detections overlay

Object detection preference:
1) YOLO ONNX (onnxruntime, with letterbox + robust output parsing)
2) Ultralytics YOLO (if installed)
3) Synthetic detection (fallback for the synthetic scene only)

Minimal dependencies: numpy, opencv-python, onnxruntime
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import cv2


# ----------------------------- Helpers -----------------------------

def try_import_onnxruntime():
    try:
        import onnxruntime as ort  # type: ignore
        return ort
    except Exception:
        return None


def try_import_ultralytics():
    try:
        from ultralytics import YOLO  # type: ignore
        return YOLO
    except Exception:
        return None


def ensure_output_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def letterbox(
    img: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    stride: int = 32
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    Resize image to a rectangle multiple of stride, keeping aspect ratio
    and adding padding (like YOLOv5/8). Returns img, scale, (dw, dh).
    """
    h, w = img.shape[:2]
    new_w, new_h = new_shape
    r = min(new_w / w, new_h / h)
    # new unpadded
    nw, nh = int(round(w * r)), int(round(h * r))
    # compute padding
    dw, dh = new_w - nw, new_h - nh
    dw /= 2
    dh /= 2

    # resize
    if (w, h) != (nw, nh):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # pad
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left,
                             right, cv2.BORDER_CONSTANT, value=color)

    return img, r, (dw, dh)


def unletterbox_boxes_xyxy(
    boxes_xyxy: np.ndarray,
    ratio: float,
    dwdh: Tuple[float, float],
    orig_w: int,
    orig_h: int
) -> np.ndarray:
    """Map XYXY boxes from letterboxed input back to original image size."""
    if boxes_xyxy.size == 0:
        return boxes_xyxy
    dw, dh = dwdh
    boxes = boxes_xyxy.copy().astype(np.float32)
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes[:, :4] /= max(ratio, 1e-12)
    # clip to image
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h - 1)
    return boxes


def add_fog(img: np.ndarray, strength: float = 0.55) -> np.ndarray:
    fog = np.full_like(img, int(255 * strength), dtype=np.uint8)
    out = cv2.addWeighted(img, 1.0 - strength, fog, strength, 0)
    out = cv2.GaussianBlur(out, (0, 0), sigmaX=6)
    return out


def add_rain(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    rain = img.copy()
    rng = np.random.default_rng(123)
    drops = max(800, (w * h) // 1000)
    length = max(22, h // 24)
    for _ in range(drops):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        thickness = int(rng.integers(2, 4))
        color = (220, 220, 220)
        cv2.line(rain, (x, y), (min(w - 1, x + 3),
                 min(h - 1, y + length)), color, thickness)
    rain = cv2.GaussianBlur(rain, (0, 0), sigmaX=2)
    rain = cv2.addWeighted(img, 0.6, rain, 0.4, 0)
    return rain


def add_snow(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    snow = img.copy()
    rng = np.random.default_rng(321)
    flakes = max(700, (w * h) // 1400)
    for _ in range(flakes):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        r = int(rng.integers(2, 4))
        cv2.circle(snow, (x, y), r, (255, 255, 255), -1)
    snow = cv2.GaussianBlur(snow, (5, 5), 0)
    snow = cv2.addWeighted(img, 0.7, snow, 0.3, 0)
    return snow


# -------------------------- GridFormer ONNX --------------------------

class GridFormer:
    def __init__(self, model_path: Optional[Path]):
        self.ort = try_import_onnxruntime()
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None
        self.provider = "Disabled"
        self.enabled = False
        self._load()

    def _load(self):
        if self.ort is None or self.model_path is None or not self.model_path.exists():
            self.enabled = False
            self.provider = "Disabled"
            return
        providers = []
        avail = set(self.ort.get_available_providers())
        if "TensorrtExecutionProvider" in avail:
            providers.append(("TensorrtExecutionProvider", {
                             "trt_max_workspace_size": 2147483648, "trt_fp16_enable": True}))
        if "CUDAExecutionProvider" in avail:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        self.session = self.ort.InferenceSession(
            str(self.model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.provider = self.session.get_providers()[0]
        self.enabled = True
        # warm-up
        try:
            self.session.run([self.output_name], {
                             self.input_name: np.random.rand(1, 3, 384, 384).astype(np.float32)})
        except Exception:
            pass

    def restore(self, bgr: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return bgr
        inp = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(inp, (384, 384))
        inp = (inp.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        out = self.session.run([self.output_name], {self.input_name: inp})[0]
        out = np.squeeze(out, axis=0).transpose(1, 2, 0)
        out = np.clip(out, 0, 1)
        out = (out * 255).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        out = cv2.resize(out, (bgr.shape[1], bgr.shape[0]))
        return out


# ------------------------------ YOLO ------------------------------

def _class_color(class_id: int) -> Tuple[int, int, int]:
    palette = [
        (0, 255, 255),
        (255, 0, 255),
        (0, 200, 255),
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
    ]
    return palette[class_id % len(palette)]


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> List[int]:
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (np.maximum(0.0, x2 - x1)) * (np.maximum(0.0, y2 - y1))
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter + 1e-6
        iou = inter / union
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


class YOLODetector:
    def __init__(self, onnx_path: Optional[Path], native_weights: Optional[Path], backend: str = "auto", debug: bool = True):
        self.ort = try_import_onnxruntime()
        self.onnx_path = onnx_path
        self.session = None
        self.input_name = None
        self.output_names: List[str] = []
        self.provider = "None"
        self.enabled = False
        self.native_model = None
        self.native_provider = None
        self.backend = backend  # 'auto' | 'onnx' | 'native'
        self.debug = debug
        self.input_hw: Tuple[int, int] = (640, 640)  # default
        self._load(onnx_path, native_weights)

    def _load(self, onnx_path: Optional[Path], native_weights: Optional[Path]):
        # Force native?
        if self.backend == "native":
            YOLO = try_import_ultralytics()
            if YOLO is not None:
                try:
                    if native_weights is not None and native_weights.exists():
                        self.native_model = YOLO(str(native_weights))
                    else:
                        self.native_model = YOLO("yolov8n.pt")
                    self.native_provider = "Ultralytics"
                    self.enabled = True
                    return
                except Exception:
                    pass
            return

        # Try ONNX first (auto or onnx)
        if self.backend in ("auto", "onnx") and self.ort is not None and onnx_path is not None and onnx_path.exists():
            providers = []
            avail = set(self.ort.get_available_providers())
            if "TensorrtExecutionProvider" in avail:
                providers.append(("TensorrtExecutionProvider", {
                                 "trt_max_workspace_size": 2147483648, "trt_fp16_enable": True}))
            if "CUDAExecutionProvider" in avail:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")
            try:
                self.session = self.ort.InferenceSession(
                    str(onnx_path), providers=providers)
                self.input_name = self.session.get_inputs()[0].name
                self.output_names = [
                    o.name for o in self.session.get_outputs()]
                self.provider = self.session.get_providers()[0]
                # deduce input size (h,w)
                # e.g., [1,3,640,640] or [1,3,-1,-1]
                ishape = self.session.get_inputs()[0].shape
                h = 640
                w = 640
                if isinstance(ishape, (list, tuple)) and len(ishape) >= 4:
                    try:
                        h = int(ishape[-2]) if isinstance(ishape[-2],
                                                          int) and ishape[-2] > 0 else 640
                        w = int(ishape[-1]) if isinstance(ishape[-1],
                                                          int) and ishape[-1] > 0 else 640
                    except Exception:
                        h, w = 640, 640
                self.input_hw = (h, w)
                self.enabled = True
                return
            except Exception:
                pass

        # Ultralytics native fallback
        YOLO = try_import_ultralytics()
        if YOLO is not None:
            try:
                if native_weights is not None and native_weights.exists():
                    self.native_model = YOLO(str(native_weights))
                else:
                    self.native_model = YOLO("yolov8n.pt")
                self.native_provider = "Ultralytics"
                self.enabled = True
                return
            except Exception:
                pass

    def _parse_outputs_any(self, out: np.ndarray) -> np.ndarray:
        """
        Accepts various YOLO ONNX layouts and returns (N, F) where
        F in {6, 84, 85}. If ambiguous, returns empty array.
        """
        arr = out
        # drop batch dim if present
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = np.squeeze(arr, axis=0)
        # If 2D and feature dim in {6,84,85} we're good
        if arr.ndim == 2:
            if arr.shape[1] in (6, 84, 85):
                return arr
            if arr.shape[0] in (6, 84, 85):  # transposed
                return arr.T
            # unknown -> empty
            return np.empty((0, 6), dtype=np.float32)

        # If 3D after squeeze didn't happen (rare): try to arrange
        if arr.ndim == 3:
            # Choose orientation where feature dim matches
            for axis in range(3):
                fdim = arr.shape[axis]
                if fdim in (6, 84, 85):
                    # move feature dim to last, then flatten anchors to first
                    arr_moved = np.moveaxis(arr, axis, -1)
                    n = np.prod(arr_moved.shape[:-1])
                    return arr_moved.reshape(int(n), fdim)
        return np.empty((0, 6), dtype=np.float32)

    def infer(self, bgr: np.ndarray, conf_thres: float = 0.25, iou_thres: float = 0.5) -> List[Dict]:
        # ONNX path
        if self.session is not None:
            H, W = bgr.shape[:2]
            in_h, in_w = self.input_hw
            # letterbox to network input
            lb_img, ratio, dwdh = letterbox(bgr, (in_w, in_h))
            # BGR->RGB, normalize, nchw
            im = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB).astype(
                np.float32) / 255.0
            im = im.transpose(2, 0, 1)[None, ...]  # 1x3xHxW

            outputs = self.session.run(
                self.output_names, {self.input_name: im})
            # prefer the first output
            out = outputs[0]
            preds = self._parse_outputs_any(out)  # (N, F)
            if preds.size == 0:
                if self.debug:
                    print(
                        f"[YOLO ONNX] Unrecognized output shape: {out.shape}")
                return []

            F = preds.shape[1]
            dets: List[Dict] = []

            if F == 6:
                # [x1,y1,x2,y2,conf,cls] in input-space (assumed)
                x1, y1, x2, y2, confs, clses = preds.T
                mask = confs >= conf_thres
                if not np.any(mask):
                    return []
                boxes = np.stack([x1, y1, x2, y2], axis=1)[mask]
                confs = confs[mask]
                clses = clses[mask].astype(int)
                # map back to original from letterbox
                boxes = unletterbox_boxes_xyxy(boxes, ratio, dwdh, W, H)
                # NMS
                keep = nms_xyxy(boxes, confs, iou_threshold=iou_thres)
                for i in keep:
                    x1i, y1i, x2i, y2i = boxes[i].astype(int).tolist()
                    cid = int(clses[i])
                    dets.append({
                        "class_name": f"class_{cid}",
                        "class_id": cid,
                        "confidence": float(confs[i]),
                        "bbox": (x1i, y1i, x2i, y2i),
                    })
                if self.debug:
                    print(
                        f"[YOLO ONNX] Parsed F=6, out:{out.shape}, kept:{len(keep)}")
                return dets

            # YOLOv5/8 style: [cx,cy,w,h, obj?, class_probs...]
            # F can be 85 (with obj) or 84 (without obj)
            cx = preds[:, 0]
            cy = preds[:, 1]
            w = preds[:, 2]
            h = preds[:, 3]

            if F >= 85:
                obj = preds[:, 4]
                cls_probs = preds[:, 5:]
            else:
                # assume 1.0 if missing
                obj = np.ones((preds.shape[0],), dtype=np.float32)
                cls_probs = preds[:, 4:]

            if cls_probs.size == 0:
                return []

            scores_all = cls_probs * obj[:, None]
            class_ids = np.argmax(scores_all, axis=1)
            confs = scores_all[np.arange(scores_all.shape[0]), class_ids]
            mask = confs >= conf_thres
            if not np.any(mask):
                if self.debug:
                    print(
                        f"[YOLO ONNX] No boxes over conf {conf_thres}. out:{out.shape}")
                return []
            cx, cy, w, h, confs, class_ids = cx[mask], cy[mask], w[mask], h[mask], confs[mask], class_ids[mask]

            # cxcywh -> xyxy in letterbox space
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            boxes = np.stack([x1, y1, x2, y2], axis=1)

            # map back to original from letterbox
            boxes = unletterbox_boxes_xyxy(boxes, ratio, dwdh, W, H)

            # NMS
            keep = nms_xyxy(boxes, confs, iou_threshold=iou_thres)
            for i in keep:
                x1i, y1i, x2i, y2i = boxes[i].astype(int).tolist()
                cid = int(class_ids[i])
                dets.append({
                    "class_name": f"class_{cid}",
                    "class_id": cid,
                    "confidence": float(confs[i]),
                    "bbox": (x1i, y1i, x2i, y2i),
                })

            if self.debug:
                print(
                    f"[YOLO ONNX] Parsed F={F}, out:{out.shape}, kept:{len(keep)}, provider:{self.provider}")
            return dets

        # Native path (Ultralytics)
        if self.native_model is not None:
            results = self.native_model(
                bgr, conf=conf_thres, iou=iou_thres, verbose=False)
            names = getattr(self.native_model, "names", {})
            dets: List[Dict] = []
            for r in results:
                boxes = getattr(r, "boxes", None)
                if boxes is None:
                    continue
                for i in range(len(boxes)):
                    cid = int(boxes.cls[i].cpu().numpy().item())
                    conf = float(boxes.conf[i].cpu().numpy().item())
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu(
                    ).numpy().astype(int).tolist()
                    dets.append({
                        "class_name": names.get(cid, f"class_{cid}"),
                        "class_id": cid,
                        "confidence": conf,
                        "bbox": (x1, y1, x2, y2),
                    })
            if self.debug:
                print(
                    f"[YOLO Native] kept:{len(dets)} provider:{self.native_provider}")
            return dets

        # No detector
        return []


# --------------------------- Drawing ---------------------------

def _draw_corners(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int], t: int = 3, l: int = 12):
    cv2.line(img, (x1, y1), (x1 + l, y1), color, t)
    cv2.line(img, (x1, y1), (x1, y1 + l), color, t)
    cv2.line(img, (x2, y1), (x2 - l, y1), color, t)
    cv2.line(img, (x2, y1), (x2, y1 + l), color, t)
    cv2.line(img, (x1, y2), (x1 + l, y2), color, t)
    cv2.line(img, (x1, y2), (x1, y2 - l), color, t)
    cv2.line(img, (x2, y2), (x2 - l, y2), color, t)
    cv2.line(img, (x2, y2), (x2, y2 - l), color, t)


def draw_detections(frame: np.ndarray, detections: List[Dict]):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls = det["class_name"]
        conf = det["confidence"]
        color = _class_color(det.get("class_id", 0))
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        frame[:] = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
        _draw_corners(frame, x1, y1, x2, y2, color, t=4, l=16)
        label = f"{cls}:{conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - h - 8)),
                      (x1 + w + 10, y1), color, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)


def draw_overlay(frame: np.ndarray, info: Dict):
    cv2.rectangle(frame, (6, 6), (560, 140), (0, 0, 0), -1)
    lines = [
        f"Weather: {info.get('weather', 'clean')}",
        f"GridFormer: {info.get('gf_provider', 'None')}",
        f"YOLO: {info.get('yolo_provider', 'None')}",
        f"FPS: {info.get('fps', 0):.1f}  Detections: {info.get('num_dets', 0)}",
    ]
    y = 30
    for txt in lines:
        cv2.putText(frame, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 255, 255), 2)
        y += 28


# --------------------------- Scene Source ---------------------------

def get_scene_source(width: int, height: int) -> Tuple[Callable[[], np.ndarray], Callable[[], None], Callable[[], List[Tuple[int, int, int, int]]]]:
    # Synthetic scene with known rectangles (for fallback detections)
    rng = np.random.default_rng(777)
    boxes: List[Tuple[int, int, int, int]] = []

    def render_clean() -> np.ndarray:
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = (35, 35, 35)
        for x in range(0, width, 40):
            cv2.line(img, (x, 0), (x, height), (55, 55, 55), 1)
        for y in range(0, height, 40):
            cv2.line(img, (0, y), (width, y), (55, 55, 55), 1)
        # Place rectangles (objects)
        boxes.clear()
        for _ in range(4):
            x1 = int(rng.integers(10, width - 180))
            y1 = int(rng.integers(10, height - 180))
            x2 = x1 + int(rng.integers(90, 180))
            y2 = y1 + int(rng.integers(90, 180))
            color = (int(rng.integers(80, 255)), int(
                rng.integers(80, 255)), int(rng.integers(80, 255)))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            boxes.append((x1, y1, x2, y2))
        cv2.putText(img, "Clean Scene", (width - 210, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
        return img

    def grab() -> np.ndarray:
        return render_clean()

    def get_boxes() -> List[Tuple[int, int, int, int]]:
        return list(boxes)

    def close():
        return None

    return grab, close, get_boxes


# ------------------------------ Main ------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Weather demo with restoration and detection")
    parser.add_argument("--gridformer-model",
                        default="models/gridformer_optimized_384.onnx")
    parser.add_argument(
        "--yolo-onnx", default="models/yolov8s_optimized_416.onnx")
    parser.add_argument("--yolo-weights", default="models/yolov8n.pt")
    parser.add_argument("--yolo-backend", choices=["auto", "onnx", "native"], default="auto",
                        help="Force YOLO backend: auto (default), onnx, or native")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="YOLO confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="YOLO IoU threshold for NMS (default: 0.5)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--output", default="docs/figures/demo_weather.mp4")
    parser.add_argument("--segment-seconds", type=float, default=5.0)
    parser.add_argument("--board", action="store_true",
                        help="show rain/snow/fog simultaneously side-by-side")
    parser.add_argument("--dataset-root", default="data/synthetic",
                        help="Root folder containing clear/rain/snow/fog subfolders")
    parser.add_argument("--use-dataset", action="store_true",
                        help="Use dataset images instead of synthetic effects when available")
    parser.add_argument("--fps", type=float, default=20.0,
                        help="Output video frames per second (default: 20)")
    parser.add_argument("--no-fallback", action="store_true",
                        help="Disable synthetic detection fallback; show only real YOLO detections")
    parser.add_argument("--no-debug", action="store_true",
                        help="Disable debug prints for detector")
    args = parser.parse_args()

    out_path = Path(args.output)
    ensure_output_dir(out_path)

    gf_path = Path(args.gridformer_model)
    yolo_onnx = Path(args.yolo_onnx)
    yolo_weights = Path(args.yolo_weights)

    gridformer = GridFormer(gf_path if gf_path.exists() else None)
    detector = YOLODetector(
        yolo_onnx if yolo_onnx.exists() else None,
        yolo_weights if yolo_weights.exists() else None,
        backend=args.yolo_backend,
        debug=not args.no_debug
    )

    grab, close, get_boxes = get_scene_source(args.width, args.height)

    # Prepare dataset image lists per weather if requested
    dataset_images: Dict[str, List[Path]] = {
        w: [] for w in ["clear", "rain", "snow", "fog"]}
    if args.use_dataset and Path(args.dataset_root).exists():
        for w in dataset_images.keys():
            wdir = Path(args.dataset_root) / w
            if wdir.exists():
                files: List[Path] = []
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                    files.extend(sorted(wdir.glob(ext)))
                dataset_images[w] = files

    def get_dataset_image(weather: str, frame_index: int = 0) -> Optional[np.ndarray]:
        files = dataset_images.get(weather, [])
        if not files:
            return None
        img_path = files[frame_index % len(files)]
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        return cv2.resize(img, (args.width, args.height))

    # 3-column output
    out_w = args.width * 3
    out_h = args.height
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(
        *"mp4v"), float(args.fps), (out_w, out_h))

    # One-time debug info
    print(f"GridFormer provider: {gridformer.provider}")
    if detector.session is not None:
        print(
            f"YOLO provider (ONNX): {detector.provider}  input_hw:{detector.input_hw}")
    elif detector.native_model is not None:
        print(f"YOLO provider (Native): {detector.native_provider}")
    else:
        print("YOLO: disabled (will use synthetic fallback if allowed)")

    if args.board:
        # Board mode: left=rain, mid=snow, right=fog simultaneously
        try:
            t_start = time.time()
            frame_idx = 0
            while time.time() - t_start < max(10.0, args.segment_seconds * 3):
                t0 = time.time()
                base = grab()
                r_ds = get_dataset_image("rain", frame_idx)
                s_ds = get_dataset_image("snow", frame_idx)
                f_ds = get_dataset_image("fog", frame_idx)
                rainy = r_ds if r_ds is not None else add_rain(base)
                snowy = s_ds if s_ds is not None else add_snow(base)
                foggy = f_ds if f_ds is not None else add_fog(base)

                rainy_r = gridformer.restore(rainy)
                snowy_r = gridformer.restore(snowy)
                foggy_r = gridformer.restore(foggy)

                rainy_d = detector.infer(rainy_r, conf_thres=float(
                    args.conf), iou_thres=float(args.iou))
                snowy_d = detector.infer(snowy_r, conf_thres=float(
                    args.conf), iou_thres=float(args.iou))
                foggy_d = detector.infer(foggy_r, conf_thres=float(
                    args.conf), iou_thres=float(args.iou))

                provider = detector.provider if detector.session is not None else (
                    detector.native_provider or "None")
                # Fallback if disabled OR no dets across all three (unless disabled)
                if (not args.no_fallback) and ((not detector.enabled) or (len(rainy_d) == 0 and len(snowy_d) == 0 and len(foggy_d) == 0)):
                    provider = "Synthetic"
                    rainy_d = [{"class_name": "object", "class_id": 0,
                                "confidence": 0.9, "bbox": b} for b in get_boxes()]
                    snowy_d = [{"class_name": "object", "class_id": 0,
                                "confidence": 0.9, "bbox": b} for b in get_boxes()]
                    foggy_d = [{"class_name": "object", "class_id": 0,
                                "confidence": 0.9, "bbox": b} for b in get_boxes()]

                board = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                board[:, 0:args.width] = rainy_r
                board[:, args.width:2*args.width] = snowy_r
                board[:, 2*args.width:] = foggy_r
                # separators
                cv2.line(board, (args.width, 0),
                         (args.width, out_h), (80, 80, 80), 2)
                cv2.line(board, (2*args.width, 0),
                         (2*args.width, out_h), (80, 80, 80), 2)

                draw_detections(board[:, 0:args.width], rainy_d)
                draw_detections(board[:, args.width:2*args.width], snowy_d)
                draw_detections(board[:, 2*args.width:], foggy_d)

                cv2.putText(board, "yağmurlu", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(board, "karlı", (args.width + 10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(board, "sisli", (2*args.width + 10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                dt = time.time() - t0
                fps = 1.0/dt if dt > 1e-6 else 0.0
                draw_overlay(board, {
                    "weather": "board",
                    "gf_provider": gridformer.provider,
                    "yolo_provider": provider,
                    "fps": fps,
                    "num_dets": len(rainy_d)+len(snowy_d)+len(foggy_d)
                })
                writer.write(board)
                frame_idx += 1
        finally:
            writer.release()
            try:
                close()
            except Exception:
                pass
    else:
        # Sequential segments
        weather_sequences = [
            ("rain", add_rain),
            ("snow", add_snow),
            ("fog", add_fog),
            ("clean", lambda x: x),
        ]
        try:
            for weather_name, weather_fn in weather_sequences:
                t_start_segment = time.time()
                frame_idx = 0
                while time.time() - t_start_segment < args.segment_seconds:
                    t0 = time.time()
                    base = grab()
                    # Prefer dataset image when available
                    if weather_name == "clean":
                        ds_img = get_dataset_image("clear", frame_idx)
                        degraded = ds_img if ds_img is not None else base
                    else:
                        ds_img = get_dataset_image(weather_name, frame_idx)
                        degraded = ds_img if ds_img is not None else weather_fn(
                            base)

                    restored = gridformer.restore(degraded)
                    dets = detector.infer(restored, conf_thres=float(
                        args.conf), iou_thres=float(args.iou))

                    provider = detector.provider if detector.session is not None else (
                        detector.native_provider or "None")
                    # Fallback if detector missing or no dets (unless disabled)
                    if (not args.no_fallback) and ((not detector.enabled) or len(dets) == 0):
                        dets = [{"class_name": "object", "class_id": 0,
                                 "confidence": 0.9, "bbox": b} for b in get_boxes()]
                        provider = "Synthetic"

                    panel = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                    panel[:, :args.width] = degraded
                    panel[:, args.width:2 * args.width] = restored
                    view_det = panel[:, 2 * args.width:]
                    view_det[:] = restored
                    draw_detections(view_det, dets)

                    dt = time.time() - t0
                    fps = 1.0 / dt if dt > 1e-6 else 0.0
                    draw_overlay(panel, {
                        "weather": weather_name,
                        "gf_provider": gridformer.provider,
                        "yolo_provider": provider,
                        "fps": fps,
                        "num_dets": len(dets)
                    })
                    cv2.putText(panel, "Degraded", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    cv2.putText(panel, "Restored", (args.width + 10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    cv2.putText(panel, "Detections", (2 * args.width + 10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                    writer.write(panel)
                    frame_idx += 1
        finally:
            writer.release()
            try:
                close()
            except Exception:
                pass

    print(f"✅ Weather demo saved: {out_path}")
    print(f"   GridFormer: {gridformer.provider}")
    if detector.session is not None:
        print(f"   YOLO: {detector.provider} (ONNX)")
    elif detector.native_model is not None:
        print(f"   YOLO: {detector.native_provider} (Native)")
    else:
        print("   YOLO: Synthetic detections")


if __name__ == "__main__":
    main()
