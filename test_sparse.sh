#!/bin/bash
export PATH="/c/msys64/ucrt64/bin:/usr/bin:/bin"
GOL="./gol"
ARCHIVE="$1"
OUT="$2"
DURATION="${3:-5}"
FFMPEG="C:/Users/ragav/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.0.1-full_build/bin/ffmpeg.exe"

echo "Rendering $ARCHIVE -> $OUT  (${DURATION}s)"
"$GOL" render "$ARCHIVE" 3840 2160 30 sparse 100 binary 8 | \
  "$FFMPEG" -f rawvideo -pixel_format rgb24 -video_size 3840x2160 -framerate 30 \
    -i - -c:v libx264 -preset veryslow -crf 18 -pix_fmt yuv420p -t "$DURATION" -y "$OUT" 2>&1
echo "EXIT: $?"
