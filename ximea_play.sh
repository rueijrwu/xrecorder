#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <video_file.mp4>"
    exit 1
fi

# Use gst-play-1.0 which is a standard GStreamer utility
# It supports interactive keyboard controls by default:
# 'q' or 'Esc' to quit
# 'p' or 'Space' to pause/resume
# 'Left'/'Right' to seek
echo "Playing $1"
echo "Controls: [q] Quit, [space] Pause/Resume, [left/right] Seek"

gst-play-1.0 "$1"
