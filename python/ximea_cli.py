import sys
import os
import time
import argparse

build_path = os.path.abspath("./ximea_highspeed_app/build")
if not os.path.exists(build_path):
    build_path = os.path.abspath("./build")
sys.path.append(build_path)

try:
    import ximea_py
except ImportError as e:
    print(f"Error: Could not import ximea_py. Make sure it is built in {build_path}")
    print(f"Details: {e}")
    sys.exit(1)

def error_handler(message):
    print(f"\n[PYTHON CALLBACK] Fatal Error Detected: {message}")

def run_cli():
    parser = argparse.ArgumentParser(description="XIMEA High-Speed Capture Python CLI")
    parser.add_argument("-o", "--output-path", default=".", help="Output directory")
    parser.add_argument("-n", "--output-name", default="capture", help="Output filename prefix")
    parser.add_argument("-f", "--save-fps", type=int, default=1000, help="Target FPS for saving")
    parser.add_argument("-d", "--display-fps", type=int, default=60, help="Target FPS for preview")
    parser.add_argument("-c", "--codec", choices=["h264", "h265", "hevc", "av1", "raw"], default="h264", help="Codec to use")

    parser.add_argument("--width", type=int, default=2048, help="Camera width")
    parser.add_argument("--height", type=int, default=1024, help="Camera height")
    parser.add_argument("-e", "--exposure", type=int, default=900, help="Exposure time (us)")
    parser.add_argument("-g", "--gain", type=float, default=-1.0, help="Gain (dB), -1 for max")
    parser.add_argument("--offset-x", type=int, default=0, help="ROI X offset")
    parser.add_argument("--offset-y", type=int, default=1040, help="ROI Y offset")

    args = parser.parse_args()
    manager = ximea_py.XimeaManager()

    cam_cfg = ximea_py.CameraConfig()
    cam_cfg.width = args.width
    cam_cfg.height = args.height
    cam_cfg.exposure_us = args.exposure
    cam_cfg.gain_db = args.gain
    cam_cfg.offset_x = args.offset_x
    cam_cfg.offset_y = args.offset_y

    print("Initializing XIMEA High-Speed Capture App (Python)...")
    print(f"Output Path: {args.output_path}, Name Prefix: {args.output_name}")
    print(f"Codec: {args.codec}, Save FPS: {args.save_fps}, Display FPS: {args.display_fps}")
    print(f"Camera: {cam_cfg.width}x{cam_cfg.height} @ {args.save_fps}Hz, ROI offset=({cam_cfg.offset_x},{cam_cfg.offset_y}), Exposure: {cam_cfg.exposure_us}us, Gain: {cam_cfg.gain_db}dB")

    if not manager.initialize(args.save_fps, args.display_fps, args.codec, cam_cfg):
        print("Failed to initialize camera. Check permissions and connection.")
        return

    if not manager.start(args.output_path, args.output_name, error_handler):
        print("Failed to start capture loop.")
        return

    print("Preview started. Press 'r' in window to toggle recording, 'q' or Ctrl+C to quit.\n")

    try:
        while manager.is_running():
            tel = manager.get_telemetry()
            rec_status = "[RECORDING]" if manager.is_recording() else "[IDLE]     "
            idx = manager.get_recording_index()
            status = (
                f"\r{rec_status} #{idx} | [CAM] Cap: {tel.cam_captured} | Drop: {tel.cam_dropped} | TO: {tel.cam_timeouts} "
                f"| [REC] Enc: {tel.rec_encoded} | Drop: {tel.rec_dropped} | Q: {tel.rec_queue} "
                f"| [DISP] {tel.prev_displayed}"
            )
            print(f"{status: <100}", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    print("\n\nCapture finished.")
    manager.stop()

if __name__ == "__main__":
    run_cli()
