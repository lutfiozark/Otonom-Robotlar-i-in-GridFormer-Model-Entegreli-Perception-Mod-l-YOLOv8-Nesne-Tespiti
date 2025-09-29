#!/usr/bin/env python3
"""Create demo GIF from RViz screenshots or video recording."""

import cv2
import numpy as np
import argparse
from pathlib import Path
import subprocess
import time
import os


def create_gif_from_images(image_dir, output_path, fps=10, duration=None):
    """Create GIF from a directory of images."""
    print(f"🎬 Creating GIF from images in {image_dir}")

    image_files = sorted(list(Path(image_dir).glob('*.png')) +
                         list(Path(image_dir).glob('*.jpg')))

    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return False

    print(f"📸 Found {len(image_files)} images")

    # Load first image to get dimensions
    first_img = cv2.imread(str(image_files[0]))
    height, width = first_img.shape[:2]

    # Create video writer (temp MP4)
    temp_video = output_path.with_suffix('.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(temp_video), fourcc, fps, (width, height))

    # Add images to video
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is not None:
            out.write(img)

    out.release()

    # Convert MP4 to GIF using ffmpeg
    if convert_mp4_to_gif(temp_video, output_path, fps):
        temp_video.unlink()  # Remove temp video
        return True
    else:
        print(f"⚠️  GIF conversion failed, keeping MP4: {temp_video}")
        return False


def convert_mp4_to_gif(input_video, output_gif, fps=10):
    """Convert MP4 to optimized GIF using ffmpeg."""
    print(f"🔄 Converting {input_video} to {output_gif}")

    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ffmpeg not found. Please install ffmpeg.")
        print("   Windows: winget install FFmpeg")
        print("   Ubuntu: sudo apt install ffmpeg")
        return False

    # FFmpeg command for high-quality GIF
    cmd = [
        'ffmpeg', '-y',  # Overwrite output
        '-i', str(input_video),
        '-vf', f'fps={fps},scale=720:-1:flags=lanczos,palettegen',
        'palette.png'
    ]

    # Generate palette
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Palette generation failed: {result.stderr}")
        return False

    # Create GIF with palette
    cmd = [
        'ffmpeg', '-y',
        '-i', str(input_video),
        '-i', 'palette.png',
        '-filter_complex', f'fps={fps},scale=720:-1:flags=lanczos[x];[x][1:v]paletteuse',
        str(output_gif)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Clean up palette
    if Path('palette.png').exists():
        Path('palette.png').unlink()

    if result.returncode == 0:
        print(f"✅ GIF created successfully: {output_gif}")
        file_size = output_gif.stat().st_size / (1024*1024)
        print(f"📊 File size: {file_size:.1f}MB")
        return True
    else:
        print(f"❌ GIF conversion failed: {result.stderr}")
        return False


def record_rviz_demo(output_dir, duration=30):
    """Record RViz demo using screen capture."""
    print(f"📹 Recording RViz demo for {duration} seconds")
    print("   Make sure RViz is open and visible on screen")
    print("   Demo will start in 5 seconds...")

    for i in range(5, 0, -1):
        print(f"   {i}...")
        time.sleep(1)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = output_dir / 'rviz_demo.mp4'

    # Platform-specific screen recording
    if os.name == 'nt':  # Windows
        print("🎥 Starting screen recording (Windows)")
        # Using built-in Windows Game Bar (requires manual start)
        print("   Please use Windows+G to start Game Bar recording")
        print("   Or use OBS Studio for better control")
        input("   Press Enter when recording is ready...")

        print("🚀 Demo is now recording!")
        print("   Perform your warehouse navigation demo")
        print(f"   Recording will stop automatically in {duration} seconds")

        time.sleep(duration)

        print("⏹️  Recording finished!")
        print("   Please save the recording and move it to:")
        print(f"   {video_path}")

        return False  # Manual process

    else:  # Linux
        # Check if ffmpeg can record X11
        try:
            cmd = [
                'ffmpeg', '-y',
                '-f', 'x11grab',
                '-video_size', '1280x720',
                '-i', ':0.0',  # Display :0.0
                '-t', str(duration),
                '-r', '15',  # 15 FPS
                str(video_path)
            ]

            print("🚀 Demo recording started!")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"✅ Recording saved: {video_path}")
                return True
            else:
                print(f"❌ Recording failed: {result.stderr}")
                return False

        except FileNotFoundError:
            print("❌ ffmpeg not found for screen recording")
            return False


def create_synthetic_demo():
    """Create a synthetic demo using sample images."""
    print("🎨 Creating synthetic demo from sample images")

    # Create demo frames
    demo_dir = Path('docs/figures/demo_frames')
    demo_dir.mkdir(parents=True, exist_ok=True)

    # Sample warehouse scenarios
    scenarios = [
        {"weather": "clear", "objects": 2, "text": "Clear conditions - baseline"},
        {"weather": "fog", "objects": 3,
            "text": "Fog detected - GridFormer enhancing"},
        {"weather": "fog", "objects": 3, "text": "Enhanced visibility"},
        {"weather": "fog", "objects": 3, "text": "YOLO detecting objects"},
        {"weather": "fog", "objects": 3, "text": "Costmap updated"},
        {"weather": "fog", "objects": 3, "text": "Path planning around obstacles"},
        {"weather": "fog", "objects": 3, "text": "Navigation successful"},
        {"weather": "rain", "objects": 4, "text": "Rain conditions - adapting"},
        {"weather": "rain", "objects": 4, "text": "Real-time enhancement"},
        {"weather": "rain", "objects": 4, "text": "Obstacle avoidance active"},
    ]

    for i, scenario in enumerate(scenarios):
        # Create synthetic frame
        frame = create_demo_frame(scenario, i)
        frame_path = demo_dir / f'frame_{i:03d}.png'
        cv2.imwrite(str(frame_path), frame)

    # Create GIF
    gif_path = Path('docs/figures/demo_nav.gif')
    gif_path.parent.mkdir(parents=True, exist_ok=True)

    return create_gif_from_images(demo_dir, gif_path, fps=2)


def create_demo_frame(scenario, frame_idx):
    """Create a synthetic demo frame."""
    # Create 720x480 frame
    frame = np.zeros((480, 720, 3), dtype=np.uint8)

    # Background color based on weather
    weather_colors = {
        'clear': (240, 248, 255),  # Alice blue
        'fog': (200, 200, 200),    # Gray
        'rain': (180, 180, 200),   # Blue-gray
        'snow': (245, 245, 245),   # White smoke
        'storm': (169, 169, 169)   # Dark gray
    }

    color = weather_colors.get(scenario['weather'], (200, 200, 200))
    frame[:] = color

    # Add title
    title = f"Warehouse AGV Demo - {scenario['text']}"
    cv2.putText(frame, title, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Add warehouse elements
    add_warehouse_elements(frame, scenario)

    # Add weather indicator
    weather_text = f"Weather: {scenario['weather'].title()}"
    cv2.putText(frame, weather_text, (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Add objects count
    objects_text = f"Objects detected: {scenario['objects']}"
    cv2.putText(frame, objects_text, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Add timestamp
    time_text = f"Time: {frame_idx * 0.5:.1f}s"
    cv2.putText(frame, time_text, (20, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    return frame


def add_warehouse_elements(frame, scenario):
    """Add warehouse elements to demo frame."""
    h, w = frame.shape[:2]

    # Add floor
    cv2.rectangle(frame, (0, h//2), (w, h), (220, 220, 200), -1)

    # Add warehouse shelves (rectangles)
    shelf_color = (139, 69, 19)  # Brown
    cv2.rectangle(frame, (50, 150), (150, 350), shelf_color, -1)
    cv2.rectangle(frame, (550, 150), (650, 350), shelf_color, -1)

    # Add detected objects (colored cubes)
    object_colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0),
                     (255, 255, 0)]  # Red, Blue, Green, Yellow

    for i in range(scenario['objects']):
        x = 200 + i * 80
        y = 300 + (i % 2) * 30
        color = object_colors[i % len(object_colors)]

        # Draw cube
        cv2.rectangle(frame, (x, y), (x+40, y+40), color, -1)
        cv2.rectangle(frame, (x, y), (x+40, y+40), (0, 0, 0), 2)

        # Add bounding box if detected
        if scenario['weather'] != 'clear' or i < 2:
            cv2.rectangle(frame, (x-5, y-5), (x+45, y+45), (0, 255, 255), 2)

    # Add AGV path
    path_points = [(100, 400), (200, 380), (300, 360),
                   (400, 350), (500, 360), (600, 380)]
    for i in range(len(path_points)-1):
        cv2.line(frame, path_points[i], path_points[i+1], (255, 0, 255), 3)

    # Add AGV position
    agv_x = 100 + (len(path_points)-1) * 20
    cv2.circle(frame, (agv_x, 390), 15, (0, 0, 255), -1)
    cv2.putText(frame, "AGV", (agv_x-15, 385),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def main():
    parser = argparse.ArgumentParser(
        description='Create warehouse AGV demo GIF')
    parser.add_argument('--mode', choices=['record', 'images', 'synthetic'],
                        default='synthetic', help='Demo creation mode')
    parser.add_argument('--input-dir', type=str,
                        help='Input directory for images mode')
    parser.add_argument('--output', type=str, default='docs/figures/demo_nav.gif',
                        help='Output GIF path')
    parser.add_argument('--duration', type=int, default=30,
                        help='Recording duration (seconds)')
    parser.add_argument('--fps', type=int, default=10, help='GIF frame rate')

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("🎬 Warehouse AGV Demo GIF Creator")
    print("=" * 40)

    success = False

    if args.mode == 'record':
        # Record live demo
        video_dir = output_path.parent / 'recordings'
        if record_rviz_demo(video_dir, args.duration):
            video_path = video_dir / 'rviz_demo.mp4'
            success = convert_mp4_to_gif(video_path, output_path, args.fps)

    elif args.mode == 'images':
        # Create from image directory
        if not args.input_dir:
            print("❌ --input-dir required for images mode")
            return
        success = create_gif_from_images(args.input_dir, output_path, args.fps)

    elif args.mode == 'synthetic':
        # Create synthetic demo
        success = create_synthetic_demo()

    if success:
        print(f"\n✅ Demo GIF created successfully!")
        print(f"📄 Output: {output_path}")
        print(f"📊 File size: {output_path.stat().st_size / 1024:.1f}KB")
        print(f"\n🚀 Ready for README and documentation!")
    else:
        print(f"\n❌ Demo creation failed")


if __name__ == '__main__':
    main()
