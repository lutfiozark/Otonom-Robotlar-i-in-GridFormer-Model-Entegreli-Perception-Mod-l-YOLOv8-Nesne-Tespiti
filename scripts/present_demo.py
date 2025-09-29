#!/usr/bin/env python3
"""
Presentation Demo Runner

Generates a short, presentable video showcasing:
- GridFormer weather restoration (ONNX Runtime: TensorRT/CUDA/CPU)
- YOLOv8 object detection (ONNX or native Ultralytics fallback)
- Live overlays: FPS, providers, detection count

Runs without ROS 2 and works on Windows. If models are missing, falls back to
identity GridFormer and native YOLOv8n.
"""

import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import cv2


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


def ensure_output_dirs(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)


class GridFormerRunner:
    def __init__(self, model_path: Optional[Path]):
        self.model_path = model_path
        self.ort_session = None
        self.input_name = None
        self.output_name = None
        self.provider = "None"
        self.enabled = False
        self.ort = try_import_onnxruntime()
        self._load()

    def _load(self):
        if self.ort is None or self.model_path is None or not self.model_path.exists():
            self.enabled = False
            self.provider = "Disabled"
            return

        available = set(self.ort.get_available_providers())
        providers: List = []
        if "TensorrtExecutionProvider" in available:
            providers.append(
                (
                    "TensorrtExecutionProvider",
                    {
                        "trt_max_workspace_size": 2147483648,
                        "trt_fp16_enable": True,
                    },
                )
            )
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.ort_session = self.ort.InferenceSession(str(self.model_path), providers=providers)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name
        self.provider = self.ort_session.get_providers()[0]
        self.enabled = True

        # warmup
        dummy = {self.input_name: np.random.rand(1, 3, 384, 384).astype(np.float32)}
        _ = self.ort_session.run([self.output_name], dummy)

    def process(self, image_bgr: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return image_bgr

        # BGR->RGB, resize to model size
        target_w = 384
        target_h = 384
        inp = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(inp, (target_w, target_h))
        inp = (inp.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]

        out = self.ort_session.run([self.output_name], {self.input_name: inp})[0]
        out = np.squeeze(out, axis=0).transpose(1, 2, 0)
        out = np.clip(out, 0, 1)
        out = (out * 255).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        out = cv2.resize(out, (image_bgr.shape[1], image_bgr.shape[0]))
        return out


class YOLOONNXRunner:
    def __init__(self, onnx_path: Optional[Path]):
        self.onnx_path = onnx_path
        self.ort = try_import_onnxruntime()
        self.ort_session = None
        self.input_name = None
        self.output_names: List[str] = []
        self.class_names: List[str] = []
        self.provider = "None"
        self.enabled = False
        self._load()

    def _load(self):
        if self.ort is None or self.onnx_path is None or not self.onnx_path.exists():
            self.enabled = False
            return

        available = set(self.ort.get_available_providers())
        providers: List = []
        if "TensorrtExecutionProvider" in available:
            providers.append(
                (
                    "TensorrtExecutionProvider",
                    {
                        "trt_max_workspace_size": 2147483648,
                        "trt_fp16_enable": True,
                    },
                )
            )
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.ort_session = self.ort.InferenceSession(str(self.onnx_path), providers=providers)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_names = [o.name for o in self.ort_session.get_outputs()]
        self.provider = self.ort_session.get_providers()[0]
        # Class names unknown for raw ONNX; leave empty for generic labels
        self.enabled = True

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> List[int]:
        if boxes.size == 0:
            return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
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
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        return keep

    def infer(self, image_bgr: np.ndarray, conf_thres: float = 0.25, iou_thres: float = 0.5) -> List[Dict]:
        if not self.enabled:
            return []

        # Preprocess (assume 640 or 416 square)
        input_shape = self.ort_session.get_inputs()[0].shape
        imsz = 416 if len(input_shape) >= 3 and input_shape[2] == 416 else 640
        resized = cv2.resize(image_bgr, (imsz, imsz))
        normalized = resized.astype(np.float32) / 255.0
        tensor = normalized.transpose(2, 0, 1)[None, ...]

        outputs = self.ort_session.run(self.output_names, {self.input_name: tensor})
        out = outputs[0]
        # Shapes: (1, 84, N) or (1, N, 84)
        if out.ndim == 3 and out.shape[1] == 84:
            preds = np.squeeze(out, axis=0).transpose(1, 0)
        elif out.ndim == 3 and out.shape[2] == 84:
            preds = np.squeeze(out, axis=0)
        else:
            return []

        boxes_xywh = preds[:, :4]
        scores_all = preds[:, 4:]
        class_ids = np.argmax(scores_all, axis=1)
        confs = scores_all[np.arange(scores_all.shape[0]), class_ids]

        mask = confs >= conf_thres
        if not np.any(mask):
            return []
        boxes_xywh, confs, class_ids = boxes_xywh[mask], confs[mask], class_ids[mask]

        cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        keep = self._nms(boxes_xyxy, confs, iou_threshold=iou_thres)
        boxes_xyxy = boxes_xyxy[keep]
        confs = confs[keep]
        class_ids = class_ids[keep]

        dets: List[Dict] = []
        for i in range(len(keep)):
            cid = int(class_ids[i])
            cname = self.class_names[cid] if 0 <= cid < len(self.class_names) else f"class_{cid}"
            x1i, y1i, x2i, y2i = boxes_xyxy[i].astype(int).tolist()
            dets.append(
                {
                    "class_name": cname,
                    "class_id": cid,
                    "confidence": float(confs[i]),
                    "bbox": (x1i, y1i, x2i, y2i),
                }
            )
        return dets


class YOLONativeRunner:
    def __init__(self, weights: Optional[Path]):
        YOLO = try_import_ultralytics()
        if YOLO is None:
            self.model = None
            self.provider = "Unavailable"
            return
        # If weights missing, download yolov8n
        if weights is None or not weights.exists():
            self.model = YOLO("yolov8n.pt")
        else:
            self.model = YOLO(str(weights))
        self.provider = "Ultralytics"

    def infer(self, image_bgr: np.ndarray, conf_thres: float = 0.25, iou_thres: float = 0.5) -> List[Dict]:
        if self.model is None:
            return []
        results = self.model(image_bgr, conf=conf_thres, iou=iou_thres, verbose=False)
        dets: List[Dict] = []
        names = self.model.names if hasattr(self.model, "names") else {}
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].cpu().numpy().item())
                conf = float(boxes.conf[i].cpu().numpy().item())
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                dets.append(
                    {
                        "class_name": names.get(cls_id, f"class_{cls_id}"),
                        "class_id": cls_id,
                        "confidence": conf,
                        "bbox": (x1, y1, x2, y2),
                    }
                )
        return dets


def draw_overlay(frame: np.ndarray, info: Dict):
    # top-left box background
    cv2.rectangle(frame, (6, 6), (360, 120), (0, 0, 0), -1)
    text_lines = [
        f"GridFormer: {info.get('gf_provider', 'None')}",
        f"YOLO: {info.get('yolo_provider', 'None')}",
        f"FPS: {info.get('fps', 0):.1f}",
        f"Detections: {info.get('num_dets', 0)}",
    ]
    y = 28
    for line in text_lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 26
    if info.get("note"):
        cv2.putText(frame, str(info.get("note")), (12, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def draw_detections(frame: np.ndarray, detections: List[Dict]):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls = det["class_name"]
        conf = det["confidence"]
        color = (0, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{cls}:{conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - h - 6), (x1 + w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def get_frame_source(use_env: bool, width: int, height: int):
    if use_env:
        try:
            from env import WarehouseEnv  # type: ignore

            env = WarehouseEnv(render_mode="DIRECT")
            env.connect()
            env.setup_scene()

            def grab():
                return cv2.cvtColor(env.get_camera_image(add_noise=True), cv2.COLOR_RGB2BGR)

            def close():
                env.disconnect()

            return grab, close
        except Exception:
            pass

    # Fallback: synthetic scene
    rng = np.random.default_rng(123)

    def grab():
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = (25, 25, 25)
        # grid background for visual structure
        for x in range(0, width, 40):
            cv2.line(img, (x, 0), (x, height), (45, 45, 45), 1)
        for y in range(0, height, 40):
            cv2.line(img, (0, y), (width, y), (45, 45, 45), 1)
        # draw large colored rectangles simulating objects
        for _ in range(4):
            x1 = int(rng.integers(10, width - 180))
            y1 = int(rng.integers(10, height - 180))
            x2 = x1 + int(rng.integers(90, 180))
            y2 = y1 + int(rng.integers(90, 180))
            color = (int(rng.integers(80, 255)), int(rng.integers(80, 255)), int(rng.integers(80, 255)))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        # title
        cv2.putText(img, "Synthetic Scene", (width - 230, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        # add light fog/noise
        fog = 25
        img = cv2.add(img, np.full_like(img, fog))
        noise = (rng.normal(0, 6, img.shape)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img

    def close():
        return None

    return grab, close


def main():
    parser = argparse.ArgumentParser(description="Run presentation demo and export video")
    parser.add_argument("--gridformer-model", default="models/gridformer_optimized_384.onnx")
    parser.add_argument("--yolo-onnx", default="models/yolov8s_optimized_416.onnx")
    parser.add_argument("--yolo-weights", default="models/yolov8n.pt")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--duration", type=float, default=15.0, help="seconds")
    parser.add_argument("--output", default="docs/figures/demo_nav.mp4")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-env", action="store_true", help="do not use PyBullet env, use synthetic")
    parser.add_argument("--side-by-side", action="store_true", help="show original vs processed side by side")
    args = parser.parse_args()

    output_path = Path(args.output)
    ensure_output_dirs(output_path)

    gf_path = Path(args.gridformer_model)
    yolo_onnx_path = Path(args.yolo_onnx)
    yolo_weights_path = Path(args.yolo_weights)

    gridformer = GridFormerRunner(gf_path if gf_path.exists() else None)
    yolo_onnx = YOLOONNXRunner(yolo_onnx_path if yolo_onnx_path.exists() else None)
    yolo_native = None if yolo_onnx.enabled else YOLONativeRunner(yolo_weights_path)

    grab_frame, close_source = get_frame_source(not args.no_env, args.width, args.height)

    out_w = args.width * 2 if args.side_by_side else args.width
    out_h = args.height
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, 20.0, (out_w, out_h))

    t0 = time.time()
    frame_times: List[float] = []
    try:
        while time.time() - t0 < args.duration:
            t_start = time.time()
            frame = grab_frame()
            if frame is None:
                break

            restored = gridformer.process(frame)

            if yolo_onnx.enabled:
                dets = yolo_onnx.infer(restored)
                provider = yolo_onnx.provider
            else:
                dets = yolo_native.infer(restored) if yolo_native else []
                provider = yolo_native.provider if yolo_native else "None"

            # Compose output frame
            if args.side_by_side:
                out_frame = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                out_frame[:, :args.width] = frame
                out_frame[:, args.width:] = restored
                # headers
                cv2.putText(out_frame, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(out_frame, "Processed", (args.width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                # draw detections on processed half with offset
                proc_view = out_frame[:, args.width:]
                draw_detections(proc_view, dets)
            else:
                out_frame = restored.copy()
                draw_detections(out_frame, dets)

            # FPS calc
            dt = time.time() - t_start
            fps = 1.0 / dt if dt > 1e-6 else 0.0
            frame_times.append(dt)

            note = None
            if not yolo_onnx.enabled and (yolo_native is None or yolo_native.model is None):
                note = "YOLO not available (native/onnx), detections disabled"
            draw_overlay(
                out_frame,
                {
                    "gf_provider": gridformer.provider,
                    "yolo_provider": provider,
                    "fps": fps,
                    "num_dets": len(dets),
                    "note": note,
                },
            )

            if not args.headless:
                cv2.imshow("GridFormer+YOLO Demo", out_frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            writer.write(out_frame)

        # Summary frame in logs
        if frame_times:
            avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
            print(f"✅ Demo complete. Avg FPS: {avg_fps:.2f}")
            print(f"   GridFormer: {gridformer.provider}")
            print(f"   YOLO: {yolo_onnx.provider if yolo_onnx.enabled else (yolo_native.provider if yolo_native else 'None')}")
            print(f"📼 Saved video: {output_path}")
        else:
            print("⚠️ No frames captured")
    finally:
        writer.release()
        try:
            close_source()
        except Exception:
            pass
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


