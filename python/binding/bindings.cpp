#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "ximea_manager.h"

namespace py = pybind11;

PYBIND11_MODULE(ximea_py, m) {
    m.doc() = "Ximea High-Speed Capture Python Wrapper";

    py::class_<XimeaTelemetry>(m, "XimeaTelemetry")
        .def_readonly("cam_captured", &XimeaTelemetry::cam_captured)
        .def_readonly("cam_dropped", &XimeaTelemetry::cam_dropped)
        .def_readonly("cam_timeouts", &XimeaTelemetry::cam_timeouts)
        .def_readonly("rec_encoded", &XimeaTelemetry::rec_encoded)
        .def_readonly("rec_dropped", &XimeaTelemetry::rec_dropped)
        .def_readonly("rec_queue", &XimeaTelemetry::rec_queue)
        .def_readonly("prev_displayed", &XimeaTelemetry::prev_displayed)
        .def_readonly("rec_cross_gpu_frames", &XimeaTelemetry::rec_cross_gpu_frames)
        .def_readonly("rec_pending_gops", &XimeaTelemetry::rec_pending_gops)
        .def_readonly("rec_gop_gaps", &XimeaTelemetry::rec_gop_gaps)
        .def_readonly("rec_frame_gaps", &XimeaTelemetry::rec_frame_gaps)
        .def_readonly("rec_bytes_written", &XimeaTelemetry::rec_bytes_written);

    py::class_<CameraConfig>(m, "CameraConfig")
        .def(py::init<>())
        .def_readwrite("width", &CameraConfig::width)
        .def_readwrite("height", &CameraConfig::height)
        .def_readwrite("exposure_us", &CameraConfig::exposure_us)
        .def_readwrite("gain_db", &CameraConfig::gain_db)
        .def_readwrite("offset_x", &CameraConfig::offset_x)
        .def_readwrite("offset_y", &CameraConfig::offset_y)
        .def_readwrite("gpu_id", &CameraConfig::gpu_id);

    py::class_<LaneConfig>(m, "LaneConfig")
        .def(py::init<>())
        .def_readwrite("gpu_id", &LaneConfig::gpu_id)
        .def_readwrite("weight", &LaneConfig::weight);

    py::class_<RecorderConfig>(m, "RecorderConfig")
        .def(py::init<>())
        .def_readwrite("gop_size", &RecorderConfig::gop_size)
        .def_readwrite("pool_size_per_lane", &RecorderConfig::pool_size_per_lane)
        .def_readwrite("max_queue_depth", &RecorderConfig::max_queue_depth)
        .def_readwrite("lanes", &RecorderConfig::lanes);

    py::class_<XimeaManager>(m, "XimeaManager")
        .def(py::init<>())
        .def("initialize", &XimeaManager::Initialize,
             py::arg("save_fps") = 1000,
             py::arg("display_fps") = 60,
             py::arg("codec") = "h264",
             py::arg("cam_cfg") = CameraConfig(),
             py::arg("rec_cfg") = RecorderConfig(),
             "Initialize the camera with target FPS for saving, display, codec, camera, and multi-lane recorder settings.")
        .def("start", &XimeaManager::Start, py::arg("output_path"), py::arg("output_name"), py::arg("error_cb") = nullptr,
        "Start capture loop and preview. Recording is toggled via 'r' key.")
        .def("stop", &XimeaManager::Stop,
             "Stop capture and finalize video files.")
        .def("is_running", &XimeaManager::IsRunning,
             "Check if the capture loop is currently active.")
        .def("is_recording", &XimeaManager::IsRecording,
             "Check if active recording is currently ongoing.")
        .def("get_recording_index", &XimeaManager::GetRecordingIndex,
             "Get the current recording index.")
        .def("get_telemetry", &XimeaManager::GetTelemetry,
             "Get real-time acquisition and encoding telemetry.");
}
