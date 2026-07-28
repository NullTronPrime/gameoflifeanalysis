@echo off
set MSYSTEM=UCRT64
set CHERE_INVOKING=1
C:\msys64\usr\bin\bash.exe --login -c "cd '/c/Users/ragav/OneDrive/Documents/wk2/gameoflife' && ./gol render 'E:/DDISK/neatlogaccess/16k_convergence.gol' 3840 2160 30 sparse binary 8 2>gol_16k.log | '/c/Users/ragav/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.0.1-full_build/bin/ffmpeg.exe' -f rawvideo -pixel_format rgb24 -video_size 3840x2160 -framerate 30 -i - -c:v libx264 -preset superfast -crf 18 -pix_fmt yuv420p 16k_sparse.mp4 2>ffmpeg_16k.log"
