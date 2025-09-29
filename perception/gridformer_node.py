#!/usr/bin/env python3
"""
GridFormer ROS 2 Node
TensorRT accelerated image restoration for weather-degraded scenes
Fallback to ONNX Runtime if TensorRT is not available
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from typing import Optional, Tuple
import time
import os
from pathlib import Path

# Try to import TensorRT, fallback to ONNX Runtime
TENSORRT_AVAILABLE = False
ONNX_AVAILABLE = False

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    print("⚠️  TensorRT not available, will use ONNX Runtime fallback")

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    print("❌ ONNX Runtime not available")


class GridFormerNode(Node):
    """ROS 2 node for GridFormer-based image restoration"""

    def __init__(self):
        super().__init__('gridformer_node')

        # Parameters
        self.declare_parameter(
            'model_path', '/workspace/models/gridformer_trained.onnx')  # Use trained model
        self.declare_parameter('input_topic', '/camera/image_raw')
        self.declare_parameter('output_topic', '/camera/image_restored')
        self.declare_parameter('enable_benchmarking', False)

        # Get parameters
        self.model_path = self.get_parameter(
            'model_path').get_parameter_value().string_value
        self.input_topic = self.get_parameter(
            'input_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter(
            'output_topic').get_parameter_value().string_value
        self.enable_benchmarking = self.get_parameter(
            'enable_benchmarking').get_parameter_value().bool_value

        # Initialize components
        self.bridge = CvBridge()

        # Model runtime variables
        self.use_tensorrt = False
        self.use_onnx = False
        self.engine = None
        self.context = None
        self.stream = None
        self.input_binding = None
        self.output_binding = None
        self.bindings = []
        self.onnx_session = None
        self.ort_session = None  # Added for ONNX optimized models
        self.input_name = None  # Added for ONNX optimized models
        self.output_name = None  # Added for ONNX optimized models
        self.model_loaded = False  # Added for model loading status

        # Performance tracking
        self.frame_count = 0
        self.total_inference_time = 0.0
        self.last_fps_report = time.time()

        # Load model
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

        # Benchmarking timer
        if self.enable_benchmarking:
            self.timer = self.create_timer(5.0, self.report_performance)

        self.get_logger().info(f"GridFormer node initialized")
        self.get_logger().info(f"Input topic: {self.input_topic}")
        self.get_logger().info(f"Output topic: {self.output_topic}")
        self.get_logger().info(f"Model path: {self.model_path}")

    def load_model(self):
        """Load GridFormer model with TensorRT/GPU optimization."""
        try:
            # Check GPU and TensorRT availability
            gpu_available = 'CUDAExecutionProvider' in ort.get_available_providers()
            trt_available = 'TensorrtExecutionProvider' in ort.get_available_providers()
            self.get_logger().info(f'🚀 GPU Available: {gpu_available}, TensorRT: {trt_available}')
            
            # Primary: Load ONNX model with TensorRT/GPU acceleration
            if Path(self.model_path).exists():
                self.get_logger().info(f'Loading GridFormer ONNX: {self.model_path}')
                
                # Prioritize TensorRT for maximum performance
                if trt_available:
                    providers = [
                        ('TensorrtExecutionProvider', {
                            'trt_max_workspace_size': 2147483648,  # 2GB
                            'trt_fp16_enable': True,
                            'trt_max_partition_iterations': 1000,
                            'trt_min_subgraph_size': 1
                        }),
                        'CUDAExecutionProvider',
                        'CPUExecutionProvider'
                    ]
                elif gpu_available:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                else:
                    providers = ['CPUExecutionProvider']
                
                self.ort_session = ort.InferenceSession(self.model_path, providers=providers)
                
                # Get input/output info
                self.input_name = self.ort_session.get_inputs()[0].name
                self.output_name = self.ort_session.get_outputs()[0].name
                
                # Log which provider is actually being used
                used_providers = self.ort_session.get_providers()
                self.get_logger().info(f'🔥 GridFormer using: {used_providers[0]}')
                
                # Test inference to trigger TensorRT compilation
                if 'TensorrtExecutionProvider' in used_providers:
                    self.get_logger().info('🏗️  Compiling TensorRT engine (first run takes longer)...')
                    dummy_input = {self.input_name: np.random.rand(1, 3, 384, 384).astype(np.float32)}
                    _ = self.ort_session.run([self.output_name], dummy_input)
                    self.get_logger().info('✅ TensorRT engine compiled successfully!')
                
                self.model_loaded = True
                return

            # Secondary: Try alternative ONNX paths
            alternative_paths = [
                f"{self.model_path.replace('.onnx', '_optimized_384.onnx')}",
                f"{self.model_path.replace('.onnx', '_384.onnx')}",
                "models/gridformer_optimized_384.onnx",
                "models/gridformer_trained.onnx"
            ]
            
            for alt_path in alternative_paths:
                if Path(alt_path).exists():
                    self.get_logger().info(f'Loading alternative GridFormer: {alt_path}')
                    
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
                        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    else:
                        providers = ['CPUExecutionProvider']
                    
                    self.ort_session = ort.InferenceSession(alt_path, providers=providers)
                    self.input_name = self.ort_session.get_inputs()[0].name
                    self.output_name = self.ort_session.get_outputs()[0].name
                    
                    used_providers = self.ort_session.get_providers()
                    self.get_logger().info(f'🔥 Alternative GridFormer using: {used_providers[0]}')
                    
                    self.model_loaded = True
                    return

            # Fallback
            self.get_logger().error(f'No suitable GridFormer model found at: {self.model_path}')
            self.model_loaded = False

        except Exception as e:
            self.get_logger().error(f'Failed to load GridFormer model: {e}')
            self.model_loaded = False

    def load_tensorrt_engine(self) -> bool:
        """Load TensorRT engine from file"""
        try:
            # Initialize TensorRT logger
            trt_logger = trt.Logger(trt.Logger.WARNING)

            # Load engine
            with open(self.model_path, 'rb') as f:
                engine_data = f.read()

            runtime = trt.Runtime(trt_logger)
            self.engine = runtime.deserialize_cuda_engine(engine_data)

            if self.engine is None:
                self.get_logger().error("Failed to load TensorRT engine")
                return False

            # Create execution context
            self.context = self.engine.create_execution_context()

            # Create CUDA stream
            self.stream = cuda.Stream()

            # Get input/output binding information
            self.setup_tensorrt_bindings()

            self.use_tensorrt = True
            self.get_logger().info("✅ TensorRT engine loaded successfully")
            return True

        except Exception as e:
            self.get_logger().error(f"Failed to load TensorRT engine: {e}")
            return False

    def load_onnx_model(self) -> bool:
        """Load ONNX model using ONNX Runtime"""
        try:
            # Check if file exists
            if not os.path.exists(self.model_path):
                self.get_logger().error(
                    f"ONNX model file not found: {self.model_path}")
                return False

            # Create ONNX Runtime session
            providers = ['CPUExecutionProvider']
            if ort.get_available_providers() and 'CUDAExecutionProvider' in ort.get_available_providers():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                self.get_logger().info("🚀 Using CUDA acceleration for ONNX")
            else:
                self.get_logger().info("💻 Using CPU for ONNX inference")

            self.onnx_session = ort.InferenceSession(
                self.model_path, providers=providers)

            # Get model info
            input_info = self.onnx_session.get_inputs()[0]
            output_info = self.onnx_session.get_outputs()[0]

            self.get_logger().info(f"Input shape: {input_info.shape}")
            self.get_logger().info(f"Output shape: {output_info.shape}")

            self.use_onnx = True
            self.get_logger().info("✅ ONNX model loaded successfully")
            return True

        except Exception as e:
            self.get_logger().error(f"Failed to load ONNX model: {e}")
            return False

    def setup_tensorrt_bindings(self):
        """Setup input/output bindings for TensorRT engine"""
        self.bindings = []

        for i in range(self.engine.num_bindings):
            binding_name = self.engine.get_binding_name(i)
            binding_shape = self.engine.get_binding_shape(i)
            binding_dtype = trt.nptype(self.engine.get_binding_dtype(i))

            if self.engine.binding_is_input(i):
                self.input_binding = {
                    'name': binding_name,
                    'shape': binding_shape,
                    'dtype': binding_dtype,
                    'size': trt.volume(binding_shape) * binding_dtype().itemsize
                }
                # Allocate device memory for input
                self.input_binding['device_mem'] = cuda.mem_alloc(
                    self.input_binding['size'])
                self.bindings.append(int(self.input_binding['device_mem']))

            else:
                self.output_binding = {
                    'name': binding_name,
                    'shape': binding_shape,
                    'dtype': binding_dtype,
                    'size': trt.volume(binding_shape) * binding_dtype().itemsize
                }
                # Allocate device memory for output
                self.output_binding['device_mem'] = cuda.mem_alloc(
                    self.output_binding['size'])
                self.bindings.append(int(self.output_binding['device_mem']))

        self.get_logger().info(f"Input shape: {self.input_binding['shape']}")
        self.get_logger().info(f"Output shape: {self.output_binding['shape']}")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for GridFormer model (384x384 optimized)"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get target size from model input shape
        target_size = (384, 384)  # Default optimized size
        if self.ort_session:
            input_shape = self.ort_session.get_inputs()[0].shape
            if len(input_shape) == 4 and input_shape[2] != 'height':  # Static shape
                target_size = (input_shape[3], input_shape[2])

        resized = cv2.resize(rgb_image, target_size)

        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # Convert to CHW format and add batch dimension
        preprocessed = np.transpose(normalized, (2, 0, 1))
        preprocessed = np.expand_dims(preprocessed, axis=0)

        return preprocessed

    def postprocess_output(self, output: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
        """Postprocess model output to image"""
        # Remove batch dimension and convert CHW to HWC
        output = output.squeeze(0)
        output = np.transpose(output, (1, 2, 0))

        # Clip to [0, 1] and convert to uint8
        output = np.clip(output, 0, 1)
        output = (output * 255).astype(np.uint8)

        # Resize to original dimensions
        output = cv2.resize(output, (original_shape[1], original_shape[0]))

        # Convert RGB to BGR
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        return output

    def inference_tensorrt(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run inference using TensorRT engine"""
        # Copy input to device
        cuda.memcpy_htod_async(
            self.input_binding['device_mem'],
            input_tensor.ravel(),
            self.stream
        )

        # Run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )

        # Copy output from device
        output = np.empty(
            self.output_binding['shape'], dtype=self.output_binding['dtype'])
        cuda.memcpy_dtoh_async(
            output,
            self.output_binding['device_mem'],
            self.stream
        )

        # Synchronize stream
        self.stream.synchronize()

        return output

    def inference_onnx(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run inference using ONNX Runtime"""
        input_name = self.onnx_session.get_inputs()[0].name
        output = self.onnx_session.run(None, {input_name: input_tensor})
        return output[0]

    def inference(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run inference with TensorRT/GPU optimization"""
        if self.ort_session:
            try:
                # ONNX Runtime inference (TensorRT/CUDA/CPU)
                ort_inputs = {self.input_name: input_tensor}
                ort_outputs = self.ort_session.run([self.output_name], ort_inputs)
                return ort_outputs[0]
            except Exception as e:
                self.get_logger().error(f"Inference failed: {e}")
                return input_tensor
        else:
            self.get_logger().error("No inference session available")
            return input_tensor

    def image_callback(self, msg: Image):
        """Process incoming image messages"""
        try:
            start_time = time.time()

            # Convert ROS message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            original_shape = cv_image.shape[:2]

            # Only process if we have a valid model
            if not (self.use_tensorrt or self.use_onnx):
                self.get_logger().warn("No model loaded, passing through original image")
                output_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
                output_msg.header = msg.header
                self.publisher.publish(output_msg)
                return

            # Preprocess
            input_tensor = self.preprocess_image(cv_image)

            # Run inference
            output_tensor = self.inference(input_tensor)

            # Postprocess
            restored_image = self.postprocess_output(
                output_tensor, original_shape)

            # Convert back to ROS message
            output_msg = self.bridge.cv2_to_imgmsg(restored_image, "bgr8")
            output_msg.header = msg.header

            # Publish
            self.publisher.publish(output_msg)

            # Performance tracking
            inference_time = time.time() - start_time
            self.total_inference_time += inference_time
            self.frame_count += 1

            if self.enable_benchmarking and self.frame_count % 30 == 0:
                avg_fps = self.frame_count / \
                    (time.time() - self.last_fps_report + 1e-6)
                avg_latency = (self.total_inference_time /
                               self.frame_count) * 1000
                backend = "TensorRT" if self.use_tensorrt else "ONNX"
                self.get_logger().info(
                    f"Performance ({backend}) - FPS: {avg_fps:.1f}, "
                    f"Avg Latency: {avg_latency:.1f}ms"
                )

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def report_performance(self):
        """Report performance metrics"""
        if self.frame_count > 0:
            elapsed = time.time() - self.last_fps_report
            fps = self.frame_count / elapsed
            avg_latency = (self.total_inference_time / self.frame_count) * 1000
            backend = "TensorRT" if self.use_tensorrt else "ONNX" if self.use_onnx else "None"

            self.get_logger().info(
                f"📊 GridFormer Performance ({backend}) - "
                f"FPS: {fps:.2f}, "
                f"Avg Latency: {avg_latency:.1f}ms, "
                f"Processed: {self.frame_count} frames"
            )

            # Reset counters
            self.frame_count = 0
            self.total_inference_time = 0.0
            self.last_fps_report = time.time()


def main(args=None):
    """Main function"""
    rclpy.init(args=args)

    try:
        node = GridFormerNode()
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
