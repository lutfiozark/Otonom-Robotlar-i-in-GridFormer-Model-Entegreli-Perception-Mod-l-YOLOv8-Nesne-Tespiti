#!/usr/bin/env python3
"""
YOLOv8 ROS 2 Node
Object detection for warehouse automation
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict
import time
from pathlib import Path
import onnxruntime as ort
import numpy as np


class YOLOv8Node(Node):
    """ROS 2 node for YOLOv8-based object detection"""

    def __init__(self):
        super().__init__('yolov8_node')

        # Parameters
        self.declare_parameter('model_path', '/workspace/models/yolov8s.onnx')
        self.declare_parameter('input_topic', '/camera/image_restored')
        self.declare_parameter('output_topic', '/camera/image_detections')
        self.declare_parameter('confidence_threshold', 0.25)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter(
            'target_classes', ['person', 'car', 'truck', 'box', 'pallet'])
        self.declare_parameter('enable_tracking', True)

        # Get parameters
        self.model_path = self.get_parameter(
            'model_path').get_parameter_value().string_value
        self.input_topic = self.get_parameter(
            'input_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter(
            'output_topic').get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter(
            'confidence_threshold').get_parameter_value().double_value
        self.iou_threshold = self.get_parameter(
            'iou_threshold').get_parameter_value().double_value
        self.target_classes = self.get_parameter(
            'target_classes').get_parameter_value().string_array_value
        self.enable_tracking = self.get_parameter(
            'enable_tracking').get_parameter_value().bool_value

        # Initialize components
        self.bridge = CvBridge()
        self.model = None
        self.ort_session = None
        self.input_name = None
        self.output_names = None
        self.model_loaded = False
        self.use_onnx = False
        self.class_names = None

        # Performance tracking
        self.frame_count = 0
        self.total_inference_time = 0.0
        self.detection_count = 0

        # Load YOLO model
        self.load_model()

        # ROS 2 setup
        self.subscription = self.create_subscription(
            Image,
            self.input_topic,
            self.image_callback,
            10
        )

        self.publisher = self.create_publisher(
            Image,
            self.output_topic,
            10
        )

        # Performance reporting timer
        self.timer = self.create_timer(10.0, self.report_performance)

        self.get_logger().info(f"YOLOv8 node initialized")
        self.get_logger().info(f"Input topic: {self.input_topic}")
        self.get_logger().info(f"Output topic: {self.output_topic}")
        self.get_logger().info(f"Model path: {self.model_path}")
        self.get_logger().info(f"Target classes: {self.target_classes}")

    def load_model(self):
        """Load YOLO model with GPU acceleration."""
        try:
            # Check GPU availability first
            available_providers = ort.get_available_providers()
            gpu_available = 'CUDAExecutionProvider' in available_providers
            trt_available = 'TensorrtExecutionProvider' in available_providers
            self.get_logger().info(
                f'🚀 GPU Available: {gpu_available}, TensorRT: {trt_available}')

            # Primary: Load ONNX model with GPU acceleration
            if Path(self.model_path).suffix == '.onnx' and Path(self.model_path).exists():
                self.get_logger().info(
                    f'Loading ONNX YOLO model: {self.model_path}')

                # Prioritize TensorRT, then CUDA, then CPU
                if trt_available:
                    providers = [
                        ('TensorrtExecutionProvider', {
                            'trt_max_workspace_size': 2147483648,
                            'trt_fp16_enable': True
                        }),
                        'CUDAExecutionProvider',
                        'CPUExecutionProvider'
                    ]
                elif gpu_available:
                    providers = ['CUDAExecutionProvider',
                                 'CPUExecutionProvider']
                else:
                    providers = ['CPUExecutionProvider']

                self.ort_session = ort.InferenceSession(
                    self.model_path, providers=providers)

                # Get input/output info
                self.input_name = self.ort_session.get_inputs()[0].name
                self.output_names = [
                    output.name for output in self.ort_session.get_outputs()]

                # Log which provider is actually being used
                used_providers = self.ort_session.get_providers()
                self.get_logger().info(
                    f'🔥 ONNX Session using: {used_providers[0]}')

                self.model_loaded = True
                self.use_onnx = True
                # Class names fallback when native model is not used
                if self.class_names is None:
                    self.class_names = list(self.target_classes) if len(
                        self.target_classes) > 0 else []
                return

            # Secondary: Try optimized ONNX files
            optimized_paths = [
                f"{Path(self.model_path).parent}/yolov8s_optimized_416.onnx",
                f"{Path(self.model_path).parent}/yolov8s.onnx",
                f"{Path(self.model_path).parent}/yolo_416.onnx"
            ]

            for onnx_path in optimized_paths:
                if Path(onnx_path).exists():
                    self.get_logger().info(
                        f'Loading optimized ONNX: {onnx_path}')

                    if trt_available:
                        providers = [
                            ('TensorrtExecutionProvider', {
                                'trt_max_workspace_size': 2147483648,
                                'trt_fp16_enable': True
                            }),
                            'CUDAExecutionProvider',
                            'CPUExecutionProvider'
                        ]
                    elif gpu_available:
                        providers = ['CUDAExecutionProvider',
                                     'CPUExecutionProvider']
                    else:
                        providers = ['CPUExecutionProvider']
                    self.ort_session = ort.InferenceSession(
                        onnx_path, providers=providers)

                    self.input_name = self.ort_session.get_inputs()[0].name
                    self.output_names = [
                        output.name for output in self.ort_session.get_outputs()]

                    used_providers = self.ort_session.get_providers()
                    self.get_logger().info(
                        f'🔥 Optimized ONNX using: {used_providers[0]}')

                    self.model_loaded = True
                    self.use_onnx = True
                    if self.class_names is None:
                        self.class_names = list(self.target_classes) if len(
                            self.target_classes) > 0 else []
                    return

            # Fallback: Native YOLO with backend parameter
            self.get_logger().warn(
                f'ONNX not found, loading native YOLO: {self.model_path}')
            self.model = YOLO(self.model_path)

            # Warm up the model
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model(dummy_image, verbose=False)

            self.get_logger().info("✅ YOLOv8 native model loaded successfully")
            self.model_loaded = True
            self.use_onnx = False
            # Use native model class names if available
            try:
                self.class_names = self.model.names
            except Exception:
                self.class_names = list(self.target_classes) if len(
                    self.target_classes) > 0 else []
            return

        except Exception as e:
            self.get_logger().error(f'Failed to load YOLO model: {e}')
            self.model_loaded = False

    def _postprocess_onnx_detections(self, ort_outputs, original_shape) -> List[Dict]:
        """Convert ONNX Runtime raw outputs to detection dictionaries with NMS.

        Returns a list of dicts: {class_name, class_id, confidence, bbox, center, area}
        """
        try:
            output = ort_outputs[0]
            # Handle shapes: (1, 84, N) or (1, N, 84)
            if output.ndim == 3 and output.shape[1] == 84:
                # (1, 84, N) -> (N, 84)
                preds = np.squeeze(output, axis=0).transpose(1, 0)
            elif output.ndim == 3 and output.shape[2] == 84:
                # (1, N, 84) -> (N, 84)
                preds = np.squeeze(output, axis=0)
            else:
                self.get_logger().warn(
                    f'Unexpected ONNX output shape: {output.shape}')
                return []

            if preds.size == 0:
                return []

            boxes_xywh = preds[:, :4]
            scores = preds[:, 4:]

            # Convert from center x,y,w,h to x1,y1,x2,y2
            cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:,
                                                        1], boxes_xywh[:, 2], boxes_xywh[:, 3]
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

            # Select best class per detection
            class_ids = np.argmax(scores, axis=1)
            confidences = scores[np.arange(scores.shape[0]), class_ids]

            # Confidence threshold
            conf_mask = confidences >= float(self.confidence_threshold)
            boxes_xyxy = boxes_xyxy[conf_mask]
            confidences = confidences[conf_mask]
            class_ids = class_ids[conf_mask]

            if boxes_xyxy.shape[0] == 0:
                return []

            # Non-maximum suppression
            keep = self._nms(boxes_xyxy, confidences,
                             iou_threshold=float(self.iou_threshold))
            boxes_xyxy = boxes_xyxy[keep]
            confidences = confidences[keep]
            class_ids = class_ids[keep]

            # Scale boxes to original image size if needed (assume model input 640 or 416)
            # Here we assume coordinates already in input scale; drawing function uses original image directly.
            detections: List[Dict] = []
            for i in range(len(keep)):
                cid = int(class_ids[i])
                cname = None
                if isinstance(self.class_names, (list, tuple)) and cid < len(self.class_names):
                    cname = self.class_names[cid]
                if cname is None:
                    cname = f'class_{cid}'

                x1i, y1i, x2i, y2i = boxes_xyxy[i].astype(int).tolist()
                detections.append({
                    'class_name': str(cname),
                    'class_id': cid,
                    'confidence': float(confidences[i]),
                    'bbox': (x1i, y1i, x2i, y2i),
                    'center': ((x1i + x2i) // 2, (y1i + y2i) // 2),
                    'area': max(0, (x2i - x1i)) * max(0, (y2i - y1i))
                })

            return detections

        except Exception as e:
            self.get_logger().error(f'Error parsing ONNX outputs: {e}')
            return []

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> List[int]:
        """Pure NumPy NMS. Returns indices of kept boxes."""
        if boxes.size == 0:
            return []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
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

    def filter_detections(self, results) -> List[Dict]:
        """Filter detections based on target classes and confidence

        Args:
            results: YOLO detection results

        Returns:
            List of filtered detection dictionaries
        """
        filtered_detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i, box in enumerate(boxes):
                # Get class name
                class_id = int(box.cls.cpu().numpy()[0])
                class_name = self.model.names[class_id]
                confidence = float(box.conf.cpu().numpy()[0])

                # Filter by confidence and target classes
                if confidence >= self.confidence_threshold:
                    if not self.target_classes or class_name in self.target_classes:
                        # Get bounding box coordinates
                        xyxy = box.xyxy.cpu().numpy()[0]
                        x1, y1, x2, y2 = map(int, xyxy)

                        detection = {
                            'class_name': class_name,
                            'class_id': class_id,
                            'confidence': confidence,
                            'bbox': (x1, y1, x2, y2),
                            'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                            'area': (x2 - x1) * (y2 - y1)
                        }
                        filtered_detections.append(detection)

        return filtered_detections

    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw detection bounding boxes and labels on image

        Args:
            image: Input image
            detections: List of detection dictionaries

        Returns:
            Image with drawn detections
        """
        # Create a copy of the image
        output_image = image.copy()

        # Color map for different classes
        colors = {
            'person': (0, 255, 0),      # Green
            'car': (255, 0, 0),         # Blue
            'truck': (0, 0, 255),       # Red
            'box': (255, 255, 0),       # Cyan
            'pallet': (255, 0, 255),    # Magenta
        }

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']

            # Get color for this class
            color = colors.get(class_name, (128, 128, 128))  # Gray as default

            # Draw bounding box
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

            # Draw label background
            cv2.rectangle(
                output_image,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )

            # Draw label text
            cv2.putText(
                output_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # Draw center point
            center_x, center_y = det['center']
            cv2.circle(output_image, (center_x, center_y), 4, color, -1)

        # Add detection count to image
        info_text = f"Detections: {len(detections)}"
        cv2.putText(
            output_image,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )

        return output_image

    def image_callback(self, msg: Image):
        """Process incoming image messages"""
        try:
            start_time = time.time()

            # Convert ROS message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Only process if we have a valid model
            if not self.model_loaded:
                self.get_logger().warn("YOLO model not loaded, skipping frame")
                return

            # Run YOLO inference
            if self.use_onnx and self.ort_session:
                # ONNX Runtime inference with GPU/CPU
                input_shape = self.ort_session.get_inputs()[0].shape
                if input_shape[2] == 416:
                    resized = cv2.resize(cv_image, (416, 416))
                else:
                    resized = cv2.resize(cv_image, (640, 640))

                normalized = resized.astype(np.float32) / 255.0
                input_tensor = np.transpose(normalized, (2, 0, 1))
                input_tensor = np.expand_dims(input_tensor, axis=0)

                ort_inputs = {self.input_name: input_tensor}
                ort_outputs = self.ort_session.run(
                    self.output_names, ort_inputs)

                # Decode ONNX outputs into detection dicts
                detections = self._postprocess_onnx_detections(
                    ort_outputs, cv_image.shape)
            else:
                # Native YOLO inference
                results = self.model(
                    cv_image,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )
                # Filter detections from native results
            detections = self.filter_detections(results)

            # Draw detections on image
            output_image = self.draw_detections(cv_image, detections)

            # Convert back to ROS message
            output_msg = self.bridge.cv2_to_imgmsg(output_image, "bgr8")
            output_msg.header = msg.header

            # Publish
            self.publisher.publish(output_msg)

            # Performance tracking
            inference_time = time.time() - start_time
            self.total_inference_time += inference_time
            self.frame_count += 1
            self.detection_count += len(detections)

            # Log detection info
            if len(detections) > 0:
                det_info = ", ".join(
                    [f"{d['class_name']}({d['confidence']:.2f})" for d in detections])
                self.get_logger().debug(f"Detected: {det_info}")

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def report_performance(self):
        """Report performance metrics"""
        if self.frame_count > 0:
            avg_fps = self.frame_count / 10.0  # 10 second reporting interval
            avg_latency = (self.total_inference_time / self.frame_count) * 1000
            avg_detections = self.detection_count / self.frame_count

            self.get_logger().info(
                f"📊 YOLOv8 Performance - "
                f"FPS: {avg_fps:.2f}, "
                f"Avg Latency: {avg_latency:.1f}ms, "
                f"Avg Detections/frame: {avg_detections:.1f}, "
                f"Total frames: {self.frame_count}"
            )

            # Reset counters
            self.frame_count = 0
            self.total_inference_time = 0.0
            self.detection_count = 0


def main(args=None):
    """Main function"""
    rclpy.init(args=args)

    try:
        node = YOLOv8Node()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
