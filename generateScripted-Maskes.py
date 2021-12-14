#######################################################################################
# example of scripting the generation process from python, without requiring command line prompts
#######################################################################################

import sys
import gc
import copy

sys.path.append('src')

from src import CmdLineArgs, GenerationCommand
CmdLineArgs.init()

from src import Hallucinator
from src import GenerationCommands
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


spatialPrompts: GenerateJob.SpatialPromptConfig  = GenerateJob.SpatialPromptConfig()
spatialPrompts.spatial_prompts=[
    ( (255,0,0), 0.1, '''Red circle'''),
    ( (0,255,0), 0.1, '''Yellow square'''),
    ( (0,0,255), 0.1, '''green hat'''),
    ( (0,0,0), 0.1, '''blue face'''),
    ( (255,255,255), 0.1, '''white paper'''),
]

spatialPrompts.append_to_prompts = ''
#spatialPrompts.prompt_key_image = './examples/4-color-mask.png'
#spatialPrompts.prompt_key_image = './examples/4-square-mask.png'
spatialPrompts.prompt_key_image = './examples/4-square-mask-circle.png'
spatialPrompts.dilate_masks = 25

# make a generation job, with a few tweaked settings, in order to build a mp4:
#   1000 iterations
#   512x512 image resolution
#   save_seq = True, to save images in sequential order ( 1,2,3,4,5 ) even if we arent saving an image every frame
#   save_freq = 2, to save every other frame thats generated
genJob = GenerateJob.GenerationJob(hallucinatorInst, totalIterations=1000, imageSizeXY=[512,512], save_seq = True, save_freq = 2)
genJob.Initialize()

# TODO: get rid of reliance on this...
genJob.use_spatial_prompts = True


#start creating the commands we need to generate an image
cut = GenerationCommands.SetCutMethod(genJob, cut_method="squish")
genJob.AddGenerationCommandFireOnce(cut, 0)


modList = HallucinatorHelpers.CreateGenerationCommandListForMaskablePrompts(genJob, spatialPrompts)

# copy prompts and masks to cpu to stroe for later, we are going to move prompts from mask to mask
maskList = []
promptList = []

for mod in modList:   
    maskList.append(mod.sourceMaskTensor.detach().clone().cpu())    
    promptList.append(mod.prompt)
    genJob.AddGenerationCommandFireOnce(mod, 0)

rmMod = GenerationCommands.RemovePrompt(genJob, removeAll=True)
genJob.AddGenerationCommandFireOnce(rmMod, 200)

#lets re-arrange them, change prompts between mods and what not
i = 1
for idx, mod in enumerate(modList):       
    modcp = GenerationCommands.AddTextPromptWithMask(genJob, promptList[i], maskTensor=maskList[idx].detach().clone(), dilateMaskAmount=mod.dilateMaskAmount, blindfold=mod.blindfold)
    genJob.AddGenerationCommandFireOnce(modcp, 200)
    i += 1
    i = i % len( promptList )

rmMod = GenerationCommands.RemovePrompt(genJob, removeAll=True)
genJob.AddGenerationCommandFireOnce(rmMod, 400)

#lets re-arrange them, change prompts between mods and what not
i = 2
for idx, mod in enumerate(modList):       
    modcp = GenerationCommands.AddTextPromptWithMask(genJob, promptList[i], maskTensor=maskList[idx].detach().clone(), dilateMaskAmount=mod.dilateMaskAmount, blindfold=mod.blindfold)
    genJob.AddGenerationCommandFireOnce(modcp, 400)
    i += 1
    i = i % len( promptList )  


rmMod = GenerationCommands.RemovePrompt(genJob, removeAll=True)
genJob.AddGenerationCommandFireOnce(rmMod, 600)

#lets re-arrange them, change prompts between mods and what not
i = 3
for idx, mod in enumerate(modList):       
    modcp = GenerationCommands.AddTextPromptWithMask(genJob, promptList[i], maskTensor=maskList[idx].detach().clone(), dilateMaskAmount=mod.dilateMaskAmount, blindfold=mod.blindfold)
    genJob.AddGenerationCommandFireOnce(modcp, 600)
    i += 1
    i = i % len( promptList )


rmMod = GenerationCommands.RemovePrompt(genJob, removeAll=True)
genJob.AddGenerationCommandFireOnce(rmMod, 800)

#lets re-arrange them, change prompts between mods and what not
i = 4
for idx, mod in enumerate(modList):       
    modcp = GenerationCommands.AddTextPromptWithMask(genJob, promptList[i], maskTensor=maskList[idx].detach().clone(), dilateMaskAmount=mod.dilateMaskAmount, blindfold=mod.blindfold)
    genJob.AddGenerationCommandFireOnce(modcp, 800)
    i += 1
    i = i % len( promptList )



# flush, clean, and go
sys.stdout.flush()
gc.collect()

try:
    hallucinatorInst.ProcessJobFull( genJob )

    # Save final image
    genJob.SaveCurrentImage()

except KeyboardInterrupt:
    pass


