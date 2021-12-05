#######################################################################################
# example of scripting the generation process from python, without requiring command line prompts
#######################################################################################

import sys
import gc
import copy

sys.path.append('src')

from src import CmdLineArgs
CmdLineArgs.init()

from src import Hallucinator
from src import GenerationMods
from src import GenerateJob
from src import HallucinatorHelpers

from PIL import ImageFile, PngImagePlugin
ImageFile.LOAD_TRUNCATED_IMAGES = True


##################################
# start actually doing stuff here
# ################################

# make the hallucinator class, with default settings
hallucinatorInst = Hallucinator.Hallucinator()
hallucinatorInst.Initialize()



#CmdLineArgs.args.spatial_prompts=[
#    ( (255,0,0), 0.1, '''teeth beksinski'''),
#    ( (0,255,0), 0.1, '''demon gustave dore'''),
#    ( (0,0,255), 0.1, '''eggs giger'''),
#    ( (0,0,0), 0.1, '''stained glass'''),
#]
#
#CmdLineArgs.args.append_to_prompts = ''
#CmdLineArgs.args.prompt_key_image = './examples/4-color-mask.png'
#CmdLineArgs.args.dilate_masks = 10
#CmdLineArgs.args.use_spatial_prompts=True

spatialPrompts: GenerateJob.SpatialPromptConfig  = GenerateJob.SpatialPromptConfig()
spatialPrompts.spatial_prompts=[
    ( (255,0,0), 0.1, '''grinning face of evil beksinski teeth tumors'''),
    ( (0,255,0), 0.1, '''rivers of blood and bodies bowing to the gold king demon of death and bones worship sacrifice gustave dore oil on canvas'''),
    ( (0,0,255), 0.1, '''i am surround by weak and petty souls, clinging to flesh oil on canvas'''),
    ( (0,0,0), 0.1, '''rotting mechanical creature with bones and flesh, in the style of dariusz zawadzki'''),
    ( (255,255,255), 0.1, '''schizophrenic nightmare creatures'''),
]

#spatialPrompts.spatial_prompts=[
#    ( (255,0,0), 0.1, '''gentle time waves'''),
#    ( (0,255,0), 0.1, '''deep space cosmic planets and stars'''),
#    ( (0,0,255), 0.1, '''fire and lava'''),
#    ( (0,0,0), 0.1, '''x-particles colorful glass'''),
#]

spatialPrompts.append_to_prompts = 'detailed painting'
#spatialPrompts.prompt_key_image = './examples/4-color-mask.png'
#spatialPrompts.prompt_key_image = './examples/4-square-mask.png'
spatialPrompts.prompt_key_image = './examples/4-square-mask-circle.png'
spatialPrompts.dilate_masks = 25

# make a generation job, with a few tweaked settings, in order to build a mp4:
#   1400 iterations
#   512x512 image resolution
#   save_seq = True, to save images in sequential order ( 1,2,3,4,5 ) even if we arent saving an image every frame
#   save_freq = 2, to save every other frame thats generated
#
# usually, we would also want to set a prompt using prompt = "prompt goes here", but I am setting prompts with generation mods instead
#genJob = GenerateJob.GenerationJob(hallucinatorInst, totalIterations=200, imageSizeXY=[512,512], save_seq = True, save_freq = 10, spatialPromptConfig=spatialPrompts, cut_method="maskTest")
genJob = GenerateJob.GenerationJob(hallucinatorInst, totalIterations=1000, imageSizeXY=[512,512], save_seq = True, save_freq = 2, cut_method="maskTest", prompts=None)
genJob.Initialize()

# TODO: get rid of reliance on this...
genJob.use_spatial_prompts = True

modList = HallucinatorHelpers.ConvertIntoMaskablePrompts(genJob, spatialPrompts)

maskList = []
promptList = []

for mod in modList:   
    maskList.append(mod.sourceMaskTensor.detach().clone().cpu())    
    promptList.append(mod.prompt)
    genJob.AddGenerationModOneShot(mod, 0)

rmMod = GenerationMods.RemovePromptMod(genJob, removeAll=True)
genJob.AddGenerationModOneShot(rmMod, 200)

#lets re-arrange them, change prompts between mods and what not
i = 1
for idx, mod in enumerate(modList):       
    modcp = GenerationMods.AddPromptMask(genJob, promptList[i], maskTensor=maskList[idx].detach().clone(), dilateMaskAmount=mod.dilateMaskAmount, blindfold=mod.blindfold)
    genJob.AddGenerationModOneShot(modcp, 200)
    i += 1
    i = i % len( promptList )

rmMod = GenerationMods.RemovePromptMod(genJob, removeAll=True)
genJob.AddGenerationModOneShot(rmMod, 400)

#lets re-arrange them, change prompts between mods and what not
i = 2
for idx, mod in enumerate(modList):       
    modcp = GenerationMods.AddPromptMask(genJob, promptList[i], maskTensor=maskList[idx].detach().clone(), dilateMaskAmount=mod.dilateMaskAmount, blindfold=mod.blindfold)
    genJob.AddGenerationModOneShot(modcp, 400)
    i += 1
    i = i % len( promptList )  


rmMod = GenerationMods.RemovePromptMod(genJob, removeAll=True)
genJob.AddGenerationModOneShot(rmMod, 600)

#lets re-arrange them, change prompts between mods and what not
i = 3
for idx, mod in enumerate(modList):       
    modcp = GenerationMods.AddPromptMask(genJob, promptList[i], maskTensor=maskList[idx].detach().clone(), dilateMaskAmount=mod.dilateMaskAmount, blindfold=mod.blindfold)
    genJob.AddGenerationModOneShot(modcp, 600)
    i += 1
    i = i % len( promptList )


rmMod = GenerationMods.RemovePromptMod(genJob, removeAll=True)
genJob.AddGenerationModOneShot(rmMod, 800)

#lets re-arrange them, change prompts between mods and what not
i = 3
for idx, mod in enumerate(modList):       
    modcp = GenerationMods.AddPromptMask(genJob, promptList[i], maskTensor=maskList[idx].detach().clone(), dilateMaskAmount=mod.dilateMaskAmount, blindfold=mod.blindfold)
    genJob.AddGenerationModOneShot(modcp, 800)
    i += 1
    i = i % len( promptList )


############
# now, lets add modifiers to control the generation of this image:
############





# flush, clean, and go
sys.stdout.flush()
gc.collect()

try:
    hallucinatorInst.ProcessJobFull( genJob )

    # Save final image
    genJob.SaveCurrentImage()

except KeyboardInterrupt:
    pass


