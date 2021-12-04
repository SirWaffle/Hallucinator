# new mods from various notebooks:
#
# inital repo:
# https://github.com/nerdyrodent/VQGAN-CLIP
#
#
# spatial masks
# https://colab.research.google.com/drive/1B9hPy1-6qhnRL3JNusFmfyWoYvjiJ1jq?usp=sharing#scrollTo=tLw9p5Rzacso
#
#
# MSE
# https://www.reddit.com/r/bigsleep/comments/onmz5r/mse_regulized_vqgan_clip/
# https://colab.research.google.com/drive/1gFn9u3oPOgsNzJWEFmdK-N9h_y65b8fj?usp=sharing#scrollTo=wSfISAhyPmyp
#
#
# MADGRAD implementation reference
# https://www.kaggle.com/yannnobrega/vqgan-clip-z-quantize-method
#
#
# torch-optimizer info
# https://pypi.org/project/torch-optimizer/
#
#
# Originally made by Katherine Crowson (https://github.com/crowsonkb, https://twitter.com/RiversHaveWings)
# The original BigGAN+CLIP method was by https://twitter.com/advadnoun


import sys
import os
import numpy as np

# shut off tqdm log spam by uncommenting the below
from tqdm import tqdm
# from functools import partialmethod
# tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

sys.path.append('src')

from src import CmdLineArgs
CmdLineArgs.init()

from src import Hallucinator

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
    if CmdLineArgs.args.prompts and CmdLineArgs.args.use_spatial_prompts == False:
        for loss in losses:
            print( "----> " + CmdLineArgs.args.prompts[promptNum] + " - loss: " + str(loss.item()) )
            promptNum += 1

    print(" ")

    if CmdLineArgs.args.log_clip:
        hallucinatorInst.WriteLogClipResults(out)
        print(" ")

    if CmdLineArgs.args.log_mem:
        hallucinatorInst.log_torch_mem()
        print(" ")

    print(" ")
    sys.stdout.flush()

    #gc.collect()



savedImageCount = 0
bestErrorScore = 99999
def train(genJob, i):
    global bestErrorScore
    global savedImageCount

    out, lossAll, lossSum = hallucinatorInst.train(genJob, i)

    # stat updates and progress images
    with torch.inference_mode():
        if i % CmdLineArgs.args.display_freq == 0:
            checkin(i, lossAll, out)  

        if i % CmdLineArgs.args.save_freq == 0:     
            if CmdLineArgs.args.save_seq == False:
                savedImageCount = i
            else:
                savedImageCount = savedImageCount + 1

            info = PngImagePlugin.PngInfo()
            info.add_text('comment', f'{CmdLineArgs.args.prompts}')
            hallucinatorInst.ConvertToPIL(out).save( build_filename_path( CmdLineArgs.args.output_dir, str(savedImageCount).zfill(5) + CmdLineArgs.args.output) , pnginfo=info)
                            
        if CmdLineArgs.args.save_best == True:

            lossAvg = lossSum / len(lossAll)

            if bestErrorScore > lossAvg.item():
                print("saving image for best error: " + str(lossAvg.item()))
                bestErrorScore = lossAvg
                info = PngImagePlugin.PngInfo()
                info.add_text('comment', f'{CmdLineArgs.args.prompts}')
                hallucinatorInst.ConvertToPIL(out).save( build_filename_path( CmdLineArgs.args.output_dir, "lowest_error_" + CmdLineArgs.args.output), pnginfo=info)










###########################################################
# start actually doing stuff here.... process cmd line args and run
# #########################################################


print("Args: " + str(CmdLineArgs.args) )


if CmdLineArgs.args.convert_to_json_cmd:
    print("json command conversion mode should be finished, exiting")
    sys.exit()

os.makedirs(os.path.dirname(CmdLineArgs.args.output_dir), exist_ok=True)



if not CmdLineArgs.args.prompts and not CmdLineArgs.args.image_prompts:
    CmdLineArgs.args.prompts = "illustrated waffle, and a SquishBrain"

if not CmdLineArgs.args.augments:
   CmdLineArgs.args.augments = [['Af', 'Pe', 'Ji', 'Er']]
elif CmdLineArgs.args.augments == 'None':
    print("Augments set to none")
    CmdLineArgs.args.augments = []


#TODO: this all needs to be changed to command line args, or config files, or something. for now, this
## hacky mask testing shit for now
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
#### /end hacky testing for spatial prompts
CmdLineArgs.args.use_spatial_prompts = False


# Do it

hallucinatorInst = Hallucinator.Hallucinator(CmdLineArgs.args)
hallucinatorInst.Initialize()

genJob = hallucinatorInst.CreateNewGenerationJob(CmdLineArgs.args)

# write out the input noise...
out = genJob.GerCurrentImageAsPIL()
info = PngImagePlugin.PngInfo()
info.add_text('comment', f'{CmdLineArgs.args.prompts}')
out.save( build_filename_path( CmdLineArgs.args.output_dir, str(0).zfill(5) + '_seed_' + CmdLineArgs.args.output ), pnginfo=info)
del out


iteration = 0 # Iteration counter
phraseCounter = 1 # Phrase counter


sys.stdout.flush()

# clean up random junk before we start
gc.collect()

# Do it
try:
    with tqdm() as pbar:
        while True:            

            # Change text prompt
            if CmdLineArgs.args.prompt_frequency > 0:
                if iteration % CmdLineArgs.args.prompt_frequency == 0 and iteration > 0:
                    # In case there aren't enough phrases, just loop
                    if phraseCounter >= len(hallucinatorInst.all_phrases):
                        phraseCounter = 0
                    
                    pMs = []
                    CmdLineArgs.args.prompts = hallucinatorInst.all_phrases[phraseCounter]

                    # Show user we're changing prompt                                
                    print(CmdLineArgs.args.prompts)
                    
                    for prompt in CmdLineArgs.args.prompts:
                        hallucinatorInst.EmbedTextPrompt(prompt)


                    phraseCounter += 1
            
            #image manipulations before training is called, such as the zoom effect
            hallucinatorInst.OnPreTrain(genJob, iteration)

            # Training time
            train(genJob, iteration)
           
            
            # Ready to stop yet?
            if iteration == CmdLineArgs.args.max_iterations:

                hallucinatorInst.OnFinishGeneration(genJob, iteration)

                # Save final image
                out = genJob.GetCurrentImageSynthed()                                  
                img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:,:,:]
                img = np.transpose(img, (1, 2, 0))
                imageio.imwrite(build_filename_path(CmdLineArgs.args.output_dir, CmdLineArgs.args.output), np.array(img))                

                if CmdLineArgs.args.log_clip:    
                    # write one for the console
                    hallucinatorInst.WriteLogClipResults(out)
                	#write once to a file for easy grabbing outside of this script                
                    text_file = open(build_filename_path( CmdLineArgs.args.output_dir, CmdLineArgs.args.output + ".txt"), "w")
                    sys.stdout = text_file
                    hallucinatorInst.WriteLogClipResults(out)
                    sys.stdout = sys.stdout 
                    text_file.close()

                break



            iteration += 1
            pbar.update()

except KeyboardInterrupt:
    pass


