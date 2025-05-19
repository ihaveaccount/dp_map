# DP map

Before launching:

    pip install pygame pyopencl numpy pillow scipy

First run dp_map.py without parameters to find the point. 

    ./run.sh --height 720

It will open a window where you can click. Then it will create a subfolder in frames/, where the selected point will be recorded

Or run windowless rendering with the test point:

    ./run.sh --pfile testpoint.json --anim --folder testpoint

run.sh is needed to restart script in case of GPU hang.

## Parameters
 - --anim - enable animation rendering. No graphics are displayed on the screen. Frames will be written in selected folder
 - --height 1080 - vertical resolution
 - --frames 1800 - number of frames for animation
 - --folder - name of subfolder in frames/, where the point for animation lies and which will contain the final frames. By default will be created with the current date
 - --pfile - path to the file with the point, by default frames/[folder]/point.txt
 - --start - start from a specific frame, by default continues from the png number in the folder
 - --kernel - path to opencl kerne
 -  --nocalc - show only rendering log with degrees and timestamps, no calculations.
 - --vertical - make vertical video
 - m1=1, m2=1, l1=1, l2=1, g=9.81, dt=0.2, iter=5000 - pendulum parameters.

 ## create_video.sh - wrapper for ffmpeg

    create_video.sh path/to/folder
