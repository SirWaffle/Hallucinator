import sys
import os
import numpy as np
import copy

# shut off tqdm log spam by uncommenting the below
# from tqdm import tqdm
# from functools import partialmethod
# tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

sys.path.append('src')

from src import cmdLineArgs
cmdLineArgs.init()

from src import Hallucinator
from src import GenerationMods
from src import GenerateJob

import gc

import torch
import imageio

from PIL import ImageFile, PngImagePlugin
ImageFile.LOAD_TRUNCATED_IMAGES = True



def build_filename_path( outputDir, filename ):
    fullpath = outputDir
    if fullpath[-1] != '\\' and fullpath[-1] != '/':
        fullpath += '/'
    fullpath += filename
    return fullpath



@torch.inference_mode()
def checkin(i, losses, out):
    print("\n*************************************************")
    print(f'i: {i}, loss sum: {sum(losses).item():g}')
    print("*************************************************")

    promptNum = 0
    lossLen = len(losses)
    if cmdLineArgs.args.prompts and lossLen <= len(cmdLineArgs.args.prompts):
        for loss in losses:            
            print( "----> " + cmdLineArgs.args.prompts[promptNum] + " - loss: " + str(loss.item()) )
            promptNum += 1
    else:
        print("mismatch in prompt numbers and losses!")

    print(" ")

    if cmdLineArgs.args.log_clip:
        hallucinatorInst.WriteLogClipResults(out)
        print(" ")

    if cmdLineArgs.args.log_mem:
        hallucinatorInst.log_torch_mem()
        print(" ")

    print(" ")
    sys.stdout.flush()

    #gc.collect()



savedImageCount = 0
bestErrorScore = 99999
def trainCallback(genJob, iteration, curImg, lossAll, lossSum):
    global bestErrorScore
    global savedImageCount    

    # stat updates and progress images
    with torch.inference_mode():
        if iteration % cmdLineArgs.args.display_freq == 0:
            checkin(iteration, lossAll, curImg)  

        if iteration % cmdLineArgs.args.save_freq == 0:     
            if cmdLineArgs.args.save_seq == True:
                savedImageCount = savedImageCount + 1                
            else:
                savedImageCount = iteration
                

            info = PngImagePlugin.PngInfo()
            info.add_text('comment', f'{cmdLineArgs.args.prompts}')
            hallucinatorInst.ConvertToPIL(curImg).save( build_filename_path( cmdLineArgs.args.output_dir, str(savedImageCount).zfill(5) + cmdLineArgs.args.output) , pnginfo=info)
                            
        if cmdLineArgs.args.save_best == True:

            lossAvg = lossSum / len(lossAll)

            if bestErrorScore > lossAvg.item():
                print("saving image for best error: " + str(lossAvg.item()))
                bestErrorScore = lossAvg
                info = PngImagePlugin.PngInfo()
                info.add_text('comment', f'{cmdLineArgs.args.prompts}')
                hallucinatorInst.ConvertToPIL(curImg).save( build_filename_path( cmdLineArgs.args.output_dir, "lowest_error_" + cmdLineArgs.args.output), pnginfo=info)










###########################################################
# start actually doing stuff here.... process cmd line args and run
# #########################################################


print("Args: " + str(cmdLineArgs.args) )


if cmdLineArgs.args.convert_to_json_cmd:
    print("json command conversion mode should be finished, exiting")
    sys.exit()

os.makedirs(os.path.dirname(cmdLineArgs.args.output_dir), exist_ok=True)



if not cmdLineArgs.args.prompts and not cmdLineArgs.args.image_prompts:
    cmdLineArgs.args.prompts = "illustrated waffle, and a SquishBrain"

if not cmdLineArgs.args.augments:
   cmdLineArgs.args.augments = [['Af', 'Pe', 'Ji', 'Er']]
elif cmdLineArgs.args.augments == 'None':
    print("Augments set to none")
    cmdLineArgs.args.augments = []



#TODO: this all needs to be changed to command line args, or config files, or something. for now, this
## hacky mask testing shit for now
#cmdLineArgs.args.spatial_prompts=[
#    ( (255,0,0), 0.1, '''teeth beksinski'''),
#    ( (0,255,0), 0.1, '''demon gustave dore'''),
#    ( (0,0,255), 0.1, '''eggs giger'''),
#    ( (0,0,0), 0.1, '''stained glass'''),
#]
#
#cmdLineArgs.args.append_to_prompts = ''
#cmdLineArgs.args.prompt_key_image = './examples/4-color-mask.png'
#cmdLineArgs.args.dilate_masks = 10
#cmdLineArgs.args.use_spatial_prompts=True
#### /end hacky testing for spatial prompts
cmdLineArgs.args.use_spatial_prompts = False



hallucinatorInst = Hallucinator.Hallucinator(cmdLineArgs.args)
hallucinatorInst.Initialize()


#argsCopy = copy.deepcopy(cmdLineArgs.args)

genJob = hallucinatorInst.CreateNewGenerationJob(cmdLineArgs.args)

doMods = False
doMods2 = True

if doMods2 == True:
    zoomMod = GenerationMods.ImageZoomInFast(genJob, zoom_scale = 1.04)
    nextStartFrame = genJob.AddGenerationMod(zoomMod, 0, 2000, 10)

    promptMod = GenerationMods.ChangePromptMod(genJob, "beautiful waves of time in a sea detailed painting colorful")
    nextStartFrame = genJob.AddGenerationMod(promptMod, 0, 0, 1)

    promptMod = GenerationMods.ChangePromptMod(genJob, "gustave dore demon faces oil painting")
    nextStartFrame = genJob.AddGenerationMod(promptMod, 400, 0, 1)

    promptMod = GenerationMods.ChangePromptMod(genJob, "psychadelic landscape of bones illustrated")
    nextStartFrame = genJob.AddGenerationMod(promptMod, 800, 0, 1)

    promptMod = GenerationMods.ChangePromptMod(genJob, "beksinski teeth evil tumor blood detailed painting")
    nextStartFrame = genJob.AddGenerationMod(promptMod, 1200, 0, 1)

    promptMod = GenerationMods.ChangePromptMod(genJob, "crystal skull colorful x-particles octane 4k HDR")
    nextStartFrame = genJob.AddGenerationMod(promptMod, 1600, 0, 1)

if doMods == True:
    #test masking from the original image into our generated one
    nextStartFrame = 0
    modLen = 1000

    #endFrame = startFrame + modLen
    # mask from original image
    #maskMod = GenerationMods.OriginalImageMask(genJob, startIt=startFrame, endIt=endFrame, freq=3, maskPath= './examples/image-mask-square-invert.png')
    #genJob.AddImageMod(maskMod)

    #startFrame = endFrame + 1
    #endFrame = startFrame + modLen
    #rot
    #rotMod = GenerationMods.ImageRotate(genJob, startIt=startFrame, endIt=endFrame, freq = 10, angle = 10)
    #genJob.AddImageMod(rotMod)

    #startFrame = endFrame + 1
    #endFrame = startFrame + modLen
    #rot back
    #rotMod = GenerationMods.ImageRotate(genJob, startIt=startFrame, endIt=endFrame, freq = 10, angle = -10)
    #genJob.AddImageMod(rotMod)

    #image zoomer
    #zoomMod = GenerationMods.ImageZoomer(genJob, startIt=startFrame, endIt=endFrame, freq = 5, zoom_scale = 1.02)
    zoomMod = GenerationMods.ImageZoomInFast(genJob, zoom_scale = 1.04)
    nextStartFrame = genJob.AddGenerationMod(zoomMod, nextStartFrame, modLen, 5)

    #startFrame = endFrame + 1
    #endFrame = startFrame + modLen
    #zoomMod = GenerationMods.ImageZoomInFast(genJob, startIt=startFrame, endIt=endFrame, freq = 5, zoom_scale = 1.04, normalizedZoomPointX=0, normalizedZoomPointY=0)
    #genJob.AddImageMod(zoomMod)

    #startFrame = endFrame + 1
    #endFrame = startFrame + modLen
    #zoomMod = GenerationMods.ImageZoomInFast(genJob, startIt=startFrame, endIt=endFrame, freq = 5, zoom_scale = 1.04)
    #genJob.AddImageMod(zoomMod)

    #startFrame = endFrame + 1
    #endFrame = startFrame + modLen
    #zoomMod = GenerationMods.ImageZoomInFast(genJob, startIt=startFrame, endIt=endFrame, freq = 5, zoom_scale = 1.04, normalizedZoomPointX=1, normalizedZoomPointY=1)
    #genJob.AddImageMod(zoomMod)

    #startFrame = endFrame + 1
    #endFrame = startFrame + modLen
    #rot
    #rotMod = GenerationMods.ImageRotate(genJob, startIt=startFrame, endIt=endFrame, freq = 10, angle = 10)
    #genJob.AddImageMod(rotMod)

    #startFrame = endFrame + 1
    #endFrame = startFrame + modLen
    #rot back
    #rotMod = GenerationMods.ImageRotate(genJob, startIt=startFrame, endIt=endFrame, freq = 10, angle = -10)
    #genJob.AddImageMod(rotMod)


#genJob2 = hallucinatorInst.CreateNewGenerationJob(argsCopy)

# Do it
curJob = genJob
#curJob.Initialize()

# write out the input noise...
out = curJob.GerCurrentImageAsPIL()
info = PngImagePlugin.PngInfo()
info.add_text('comment', f'{cmdLineArgs.args.prompts}')
out.save( build_filename_path( cmdLineArgs.args.output_dir, str(0).zfill(5) + '_seed_' + cmdLineArgs.args.output ), pnginfo=info)
del out

# clean, flush, and go
sys.stdout.flush()
gc.collect()

try:
    hallucinatorInst.ProcessJobFull( curJob, trainCallback )

    # Save final image
    out = curJob.GerCurrentImageAsPIL()
    out.save( build_filename_path( cmdLineArgs.args.output_dir, cmdLineArgs.args.output ))               

    if cmdLineArgs.args.log_clip:    
        # write one for the console
        hallucinatorInst.WriteLogClipResults(out)
        #write once to a file for easy grabbing outside of this script                
        text_file = open(build_filename_path( cmdLineArgs.args.output_dir, cmdLineArgs.args.output + ".txt"), "w")
        sys.stdout = text_file
        hallucinatorInst.WriteLogClipResults(out)
        sys.stdout = sys.stdout 
        text_file.close()

except KeyboardInterrupt:
    pass


