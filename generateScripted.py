#######################################################################################
# example of scripting the generation process from python, without requiring command line prompts
#######################################################################################

import sys
import gc

sys.path.append('src')

from src import CmdLineArgs
CmdLineArgs.init()

from src import Hallucinator
from src import GenerationCommands
from src import GenerateJob

from PIL import ImageFile, PngImagePlugin
ImageFile.LOAD_TRUNCATED_IMAGES = True


##################################
# start actually doing stuff here
# ################################

# make the hallucinator class, with default settings
hallucinatorInst = Hallucinator.Hallucinator()
hallucinatorInst.Initialize()

# make a generation job, with a few tweaked settings, in order to build a mp4:
#   1400 iterations
#   512x512 image resolution
#   save_seq = True, to save images in sequential order ( 1,2,3,4,5 ) even if we arent saving an image every frame
#   save_freq = 2, to save every other frame thats generated
#
genJob = GenerateJob.GenerationJob(hallucinatorInst, totalIterations=1600, imageSizeXY=[512,512], save_seq = True, save_freq = 4)
genJob.Initialize()

############
# now, lets add modifiers to control the generation of this image:
############

# lets use a non default optimizert ( adam is default)
opt = GenerationCommands.SetOptimiser(genJob, optimizerName="MADGRAD", learningRate=6)
genJob.AddGenerationCommandFireOnce(opt, 0)

# define our cut method
cut = GenerationCommands.SetCutMethod(genJob, cut_method="latest")
genJob.AddGenerationCommandFireOnce(cut, 0)

# zoom for the entire duration
zoomMod = GenerationCommands.ImageZoomInFast(genJob, zoom_scale = 1.05)
genJob.AddGenerationCommand(zoomMod, 0, 1600, 10)

# change prompts every so often
promptMod = GenerationCommands.AddTextPrompt(genJob, "x-particles brain v-ray colorful wires 4k HDR")
genJob.AddGenerationCommandFireOnce(promptMod, 0)

promptMod = GenerationCommands.AddTextPrompt(genJob, "red crystal x-particles v-ray beautiful ethereal")
genJob.AddGenerationCommandFireOnce(promptMod, 200)

promptMod = GenerationCommands.AddTextPrompt(genJob, "heart x-particles v-ray blood 4k HDR")
genJob.AddGenerationCommandFireOnce(promptMod, 400)

promptMod = GenerationCommands.AddTextPrompt(genJob, "blue white glass x-particles v-ray ocean ball")
genJob.AddGenerationCommandFireOnce(promptMod, 600)

# lets rotate here for the next prompt
rotMod = GenerationCommands.ImageRotate(genJob, angle = 6)
genJob.AddGenerationCommand(rotMod, 800, 200, 10)

promptMod = GenerationCommands.AddTextPrompt(genJob, "one large human eye detailed x-particles v-ray 4k HDR")
genJob.AddGenerationCommandFireOnce(promptMod, 800)

# rotation and eye are done, move on
promptMod = GenerationCommands.AddTextPrompt(genJob, "white crystal gem growth bone x-particles v-ray")
genJob.AddGenerationCommandFireOnce(promptMod, 1000)

promptMod = GenerationCommands.AddTextPrompt(genJob, "human skull x-particles v-ray detailed 4k HDR")
genJob.AddGenerationCommandFireOnce(promptMod, 1200)

# lets finish off by returning to the original prompt, maybe we can loop if we are super lucky
promptMod = GenerationCommands.AddTextPrompt(genJob, "x-particles brain v-ray colorful wires 4k HDR")
genJob.AddGenerationCommandFireOnce(promptMod, 1400)

# flush, clean, and go
sys.stdout.flush()
gc.collect()

try:
    hallucinatorInst.ProcessJobFull( genJob )

    # Save final image
    genJob.SaveCurrentImage()

except KeyboardInterrupt:
    pass


