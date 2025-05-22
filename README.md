# DP map

Before launching:

    pip install pygame pyopencl numpy pillow scipy

First run dp_map.py without parameters to find the point. 

    ./run.sh --height 720

It will open a window where you can click. Then it will create a subfolder in frames/, where the selected point will be recorded

Or run windowless rendering with the test point:

    ./run.sh --pfile testpoint.json --anim --folder point.json

run.sh is needed to restart script in case of GPU hang.

## Parameters
 - --anim - enable zoom animation rendering. No graphics are displayed on the screen. Frames will be written in selected folder
 - --paranim - parameter change animation, without zoom
 - --height - vertical resolution. Default: 1080
 - --frames - number of frames for animation. Default: 1800
 - --folder - name of subfolder in frames/, where the point for animation lies and which will contain the final frames. Default: created with the current date.
 - --pfile - path to the file with the point.  Default: frames/[folder]/point.json
 - --start - start from a specific frame. Default: continues from the png number in the folder
 - --kernel - shader filename. Default: pendulum.c
 - --skipcalc - show only rendering log with degrees and timestamps, no calculations.
 - --vertical - swap width and height.
 - --median - median filter size for antialiasing. Default: 3.
 - --invert - invert colors
 - --x_min, x_max, y_min, y_max - starting view params. Default: kernel settings.
 - --param PARAM_NAME PARAM_VALUE - set params according to shader values. Default: kernel settings.
 
 ## create_video.sh - wrapper for ffmpeg

    ./create_video.sh path/to/folder
