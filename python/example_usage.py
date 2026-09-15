import sys
import os
import time

# 1. Add the build directory to the path so Python can find 'ximea_py.so'
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
    """Callback function triggered by the C++ core on fatal errors."""
    print(f"\n[PYTHON CALLBACK] Fatal Error Detected: {message}")

def run_example():
    # 2. Instantiate the Manager
    manager = ximea_py.XimeaManager()

    # 3. Initialize Settings
    save_fps = 1000
    display_fps = 60
    codec = "h264" 

    # 3b. Configure Camera
    cam_cfg = ximea_py.CameraConfig()
    cam_cfg.width = 4096
    cam_cfg.height = 996
    cam_cfg.exposure_us = 900
    cam_cfg.gain_db = -1.0 # -1 means max analog gain
    cam_cfg.offset_x = 0
    cam_cfg.offset_y = 0

    print(f"--- Initializing XIMEA Manager ---")
    print(f"Target: {save_fps} FPS | Preview: {display_fps} FPS | Codec: {codec}")
    print(f"Camera: {cam_cfg.width}x{cam_cfg.height}, Exposure: {cam_cfg.exposure_us}us, Gain: {cam_cfg.gain_db}dB")
    
    if not manager.initialize(save_fps, display_fps, codec, cam_cfg):
        print("Failed to initialize camera. Check permissions and connection.")
        return

    # 4. Start Acquisition
    output_dir = "."
    file_prefix = "python_highspeed_test"
    
    print(f"--- Starting Capture ---")
    if not manager.start(output_dir, file_prefix, error_handler):
        print("Failed to start capture loop.")
        return

    print("Capture running. Monitoring telemetry...")
    print("Press 'r' in the preview window to toggle recording.")
    print("Press 'q' in the preview window or Ctrl+C to quit.\n")

    try:
        while manager.is_running():
            # 5. Fetch Real-Time Telemetry
            tel = manager.get_telemetry()
            rec_status = "[RECORDING]" if manager.is_recording() else "[IDLE]"
            idx = manager.get_recording_index()
            
            # Print a live status line
            status = (
                f"\r{rec_status} #{idx} | [CAM] Cap: {tel.cam_captured} | Drop: {tel.cam_dropped} "
                f"| [REC] Enc: {tel.rec_encoded} | Q: {tel.rec_queue} "
                f"| [DISP] {tel.prev_displayed}"
            )
            print(f"{status: <100}", end="", flush=True)
            
            time.sleep(0.1) # Update at 10Hz
            
    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    # 6. Graceful Stop
    print(f"\n\n--- Stopping Capture ---")
    manager.stop()
    print(f"Capture finished. Video saved to {file_prefix}.mkv")

if __name__ == "__main__":
    run_example()
