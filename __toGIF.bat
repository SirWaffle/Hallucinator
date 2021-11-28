REM ffmpeg -f image2 -framerate 1 -i simpimgs%03d.jpg -loop -1 simpson.gif
cd output
REM D:\Programming\generativeArt\ffmpeg\bin\ffmpeg.exe -f image2 -framerate 2 -pattern_type glob -i './output/*.png' ./output.gif

REM !/bin/sh
REM palette="/tmp/palette.png"
REM filters="fps=15,scale=320:-1:flags=lanczos"
REM ffmpeg -v warning -i $1 -vf "$filters,palettegen=stats_mode=diff" -y $palette
REM ffmpeg -i $1 -i $palette -lavfi "$filters,paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" -y $2

D:\Programming\generativeArt\ffmpeg\bin\ffmpeg.exe -f image2 -framerate 10 -i %%05doutput.png __output.gif


REM or video
REM ffmpeg -framerate 1/2 -i img%04d.png -c:v libx264 -r 30 out.mp4

REM resulted in blackscreen
D:\Programming\generativeArt\ffmpeg\bin\ffmpeg.exe -framerate 30 -i %%05doutput.png -c:v libx264 __output.mp4

REM D:\Programming\generativeArt\ffmpeg\bin\ffmpeg.exe -framerate 30 -pix_fmt yuv420p -i %%05doutput.png __output.mp4

PAUSE