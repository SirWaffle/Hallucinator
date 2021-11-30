REM ffmpeg -f image2 -framerate 1 -i simpimgs%03d.jpg -loop -1 simpson.gif

set START_FRAME=-start_number 15

REM frameratestuff
SET /A INTERPOLATE_TO_FRAMERATE=48
SET /A framerate=24
SET compressedFramerate=fps=14


REM scale
SET filters=scale=iw:ih
REM SET filters=scale=iw*0.5:ih*0.5
REM SET filters=scale=300:300
REM SET filters=scale='min(512,iw)':'min(512,ih)'

REM motion interpolated frames
REM SET filters=%filters%,minterpolate=fps=%INTERPOLATE_TO_FRAMERATE%:mi_mode=blend
REM SET filters=%filters%,minterpolate=fps=%INTERPOLATE_TO_FRAMERATE%:mi_mode=mci

SET MAKE_GIF=1
SET MAKE_COMPRESSED_GIF=0
SET MAKE_MP4=1


cd output

if %MAKE_GIF% == 1 (
	REM uncompressed gif
	D:\Programming\generativeArt\ffmpeg\bin\ffmpeg.exe -f image2 %START_FRAME% -framerate %framerate% -lavfi "%filters%" -i %%05doutput.png -y __output.gif
)

if %MAKE_COMPRESSED_GIF% == 1 (
	REM lets try with pallete gen gif
	REM D:\Programming\generativeArt\ffmpeg\bin\ffmpeg.exe -f image2 -i %%05doutput.png -vf "palettegen=stats_mode=diff" -y __palette.png
	REM D:\Programming\generativeArt\ffmpeg\bin\ffmpeg.exe -f image2 -framerate %framerate%  -i %%05doutput.png -i __palette.png -lavfi "%filters%,fps=12,paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" -y __outputPalette.gif
	
	REM just use a lower framerate for now
	D:\Programming\generativeArt\ffmpeg\bin\ffmpeg.exe -f image2 -framerate %framerate% -lavfi "%filters%,%compressedFramerate%" -i %%05doutput.png -y __outputLowFR.gif
)

if %MAKE_MP4% == 1 (
	REM or video
	D:\Programming\generativeArt\ffmpeg\bin\ffmpeg.exe -framerate %framerate% -lavfi "%filters%" -i %%05doutput.png -c:v libx264 -y __output.mp4
)


PAUSE