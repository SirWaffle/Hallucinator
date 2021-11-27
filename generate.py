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

from src import cmdLineArgs
cmdLineArgs.init()

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
    if cmdLineArgs.args.prompts and cmdLineArgs.args.use_spatial_prompts == False:
        for loss in losses:
            print( "----> " + cmdLineArgs.args.prompts[promptNum] + " - loss: " + str(loss.item()) )
            promptNum += 1

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




bestErrorScore = 99999
def train(i):
    global bestErrorScore
      
    out, lossAll, lossSum = hallucinatorInst.train(i)

    # stat updates and progress images
    with torch.inference_mode():
        if i % cmdLineArgs.args.display_freq == 0:
            checkin(i, lossAll, out)  

        if i % cmdLineArgs.args.save_freq == 0:          
            info = PngImagePlugin.PngInfo()
            info.add_text('comment', f'{cmdLineArgs.args.prompts}')
            hallucinatorInst.ConvertToPIL(out).save( build_filename_path( cmdLineArgs.args.output_dir, str(i).zfill(5) + cmdLineArgs.args.output) , pnginfo=info)
                            
        if cmdLineArgs.args.save_best == True:

            lossAvg = lossSum / len(lossAll)

            if bestErrorScore > lossAvg.item():
                print("saving image for best error: " + str(lossAvg.item()))
                bestErrorScore = lossAvg
                info = PngImagePlugin.PngInfo()
                info.add_text('comment', f'{cmdLineArgs.args.prompts}')
                hallucinatorInst.ConvertToPIL(out).save( build_filename_path( cmdLineArgs.args.output_dir, "lowest_error_" + cmdLineArgs.args.output), pnginfo=info)










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


##### lets hard code a test for locking parts of the image with a mask
#cmdLineArgs.args.image_lock_mask = './examples/image-mask-square.png'
#cmdLineArgs.args.image_lock_mask = './examples/image-mask-square-invert.png'
cmdLineArgs.args.use_image_lock_mask = False
#cmdLineArgs.args.image_lock_overwrite_iteration = 3 #3 actually seems to work allright... this code needs a massive speedup though



# Do it

hallucinatorInst = Hallucinator.Hallucinator(cmdLineArgs.args)
hallucinatorInst.FullInitialize()

# write out the input noise...
out = hallucinatorInst.GerCurrentImageAsPIL()
info = PngImagePlugin.PngInfo()
info.add_text('comment', f'{cmdLineArgs.args.prompts}')
out.save( build_filename_path( cmdLineArgs.args.output_dir, str(0).zfill(5) + '_seed_' + cmdLineArgs.args.output ), pnginfo=info)
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
            if cmdLineArgs.args.prompt_frequency > 0:
                if iteration % cmdLineArgs.args.prompt_frequency == 0 and iteration > 0:
                    # In case there aren't enough phrases, just loop
                    if phraseCounter >= len(hallucinatorInst.all_phrases):
                        phraseCounter = 0
                    
                    pMs = []
                    cmdLineArgs.args.prompts = hallucinatorInst.all_phrases[phraseCounter]

                    # Show user we're changing prompt                                
                    print(cmdLineArgs.args.prompts)
                    
                    for prompt in cmdLineArgs.args.prompts:
                        hallucinatorInst.EmbedTextPrompt(prompt)


                    phraseCounter += 1
            
            #image manipulations before training is called, such as the zoom effect
            hallucinatorInst.OnPreTrain(iteration)

            # Training time
            train(iteration)
           
            
            # Ready to stop yet?
            if iteration == cmdLineArgs.args.max_iterations:

                hallucinatorInst.OnFinishGeneration()

                # Save final image
                out = hallucinatorInst.GetCurrentImageSynthed()                                  
                img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:,:,:]
                img = np.transpose(img, (1, 2, 0))
                imageio.imwrite(build_filename_path(cmdLineArgs.args.output_dir, cmdLineArgs.args.output), np.array(img))                

                if cmdLineArgs.args.log_clip:    
                    # write one for the console
                    hallucinatorInst.WriteLogClipResults(out)
                	#write once to a file for easy grabbing outside of this script                
                    text_file = open(build_filename_path( cmdLineArgs.args.output_dir, cmdLineArgs.args.output + ".txt"), "w")
                    sys.stdout = text_file
                    hallucinatorInst.WriteLogClipResults(out)
                    sys.stdout = sys.stdout 
                    text_file.close()

                break



            iteration += 1
            pbar.update()

except KeyboardInterrupt:
    pass


