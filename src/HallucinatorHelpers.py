from typing import List

from PIL import Image
import numpy as np
import torch
from src import GenerationMods, Hallucinator
from src import GenerateJob
from torchvision.transforms import functional as TF

###########################################
#
# Some helper methods for creating isntances of hallucinator from command line args
# and some other default behaviors so we can quickyl make scripts
#
###########################################


# all command line args pushed to class. clean this up once global use of argparse values is cleaned up

def CreateHallucinatorFromArgParse( args ) -> Hallucinator.Hallucinator:
    hallucinatorInst = Hallucinator.Hallucinator( clipModel = args.clip_model,vqgan_config_path=args.vqgan_config, 
                                              vqgan_checkpoint_path=args.vqgan_checkpoint, use_mixed_precision=args.use_mixed_precision,
                                              clip_cpu=args.clip_cpu, cuda_device=args.cuda_device, anomaly_checker = args.anomaly_checker,
                                              deterministic = args.deterministic, log_clip = args.log_clip, log_clip_oneshot = args.log_clip_oneshot, 
                                              log_mem = args.log_mem, display_freq = args.display_freq )
    return hallucinatorInst




def CreateGenerationJobFromArgParse( hallucinatorInst:Hallucinator.Hallucinator, args ) -> GenerateJob.GenerationJob:    
    genJob = GenerateJob.GenerationJob( hallucinatorInst, cut_method = args.cut_method, totalIterations = args.max_iterations, prompts = args.prompts,
                                        image_prompts = args.image_prompts, startingImage = args.init_image, imageSizeXY = args.size, 
                                        cutNum=args.cutn, cutSize=args.cut_size, cutPow=args.cut_pow, augments=args.augments,
                                        optimizerName=args.optimizer,init_weight=args.init_weight, init_noise=args.init_noise,
                                        noise_prompt_seeds=args.noise_prompt_seeds, noise_prompt_weights=args.noise_prompt_weights, prompt_frequency=args.prompt_frequency,
                                        deterministic = args.deterministic, outputDir = args.output_dir, outputFilename = args.output, save_freq = args.save_freq,
                                        save_seq = args.save_seq, save_best = args.save_best)
    return genJob



# TODO: figure out blindfolding stuff
def ConvertIntoMaskablePrompts(genJob:GenerateJob.GenerationJob, spatialPromptConfig:GenerateJob.SpatialPromptConfig) -> List[GenerationMods.AddPromptMask]:
    #Make prompt masks
    img = Image.open(spatialPromptConfig.prompt_key_image)
    pil_image = img.convert('RGB')

    prompt_key_image = np.asarray(pil_image)

    #Set up color->prompt map
    color_to_prompt_idx={}
    all_prompts=[]
    blindfolds=[]
    for i,(color_key,blind,prompt) in enumerate(spatialPromptConfig.spatial_prompts):
        #append a collective promtp to all, to keep a set style if we want
        if prompt[-1]==' ':
            prompt+= spatialPromptConfig.append_to_prompts
        elif prompt[-1]=='.' or prompt[-1]=='|' or prompt[-1]==',':
            prompt+=" "+spatialPromptConfig.append_to_prompts
        else:
            prompt+=". "+spatialPromptConfig.append_to_prompts

        all_prompts.append(prompt)
        blindfolds.append(blind)
        color_to_prompt_idx[color_key] = i
    
    color_to_prompt_idx_orig = dict(color_to_prompt_idx)

    #init the masks
    prompt_masks = torch.FloatTensor(
        len(spatialPromptConfig.spatial_prompts),
        1, #color channel
        prompt_key_image.shape[0],
        prompt_key_image.shape[1]).fill_(0)

    #go pixel by pixel and assign it to one mask, based on closest color
    for y in range(prompt_key_image.shape[0]):
        for x in range(prompt_key_image.shape[1]):
            key_color = tuple(prompt_key_image[y,x])

            if key_color not in color_to_prompt_idx:
                min_dist=999999
                best_idx=-1
                for color,idx in color_to_prompt_idx_orig.items():
                    dist = abs(color[0]-key_color[0])+abs(color[1]-key_color[1])+abs(color[2]-key_color[2])
                    #print('{} - {} = {}'.format(color,key_color,dist))
                    if dist<min_dist:
                        min_dist = dist
                        best_idx=idx
                color_to_prompt_idx[key_color]=best_idx #store so we don't need to compare again
                idx = best_idx
            else:
                idx = color_to_prompt_idx[key_color]

            prompt_masks[idx,0,y,x]=1

    #prompt_masks = prompt_masks.to(self.vqganDevice)

    #todo, create prompt mod things here
    modList:List[GenerationMods.AddPromptMask] = []
    maskIdx: int = 0
    for prompt in all_prompts:
        mask = prompt_masks[maskIdx]
        blindfold = blindfolds[maskIdx]
        promptMod = GenerationMods.AddPromptMask(genJob, prompt, maskTensor=mask, dilateMaskAmount=spatialPromptConfig.dilate_masks, blindfold=blindfold)
        modList.append( promptMod)

        maskIdx += 1

    #rough display
    if prompt_masks.size(0)>=4:
        print('first 3 masks')
        TF.to_pil_image(prompt_masks[0,0].detach().cpu()).save('ex-masks-0.png')   
        TF.to_pil_image(prompt_masks[1,0].detach().cpu()).save('ex-masks-1.png')
        TF.to_pil_image(prompt_masks[2,0].detach().cpu()).save('ex-masks-2.png')
        TF.to_pil_image(prompt_masks[3,0].detach().cpu()).save('ex-masks-3.png')
        TF.to_pil_image(prompt_masks[0:4,0].detach().cpu()).save('ex-masks-comb.png')
        #display.display(display.Image('ex-masks.png')) 
        if prompt_masks.size(0)>=6:
            print('next 3 masks')
            TF.to_pil_image(prompt_masks[3:6,0].detach().cpu()).save('ex-masks.png') 
            #display.display(display.Image('ex-masks.png')) 
    

    return modList

    '''
    if any(self.blindfold):
        #Set up blur used in blindfolding
        k=13
        self.blur_conv = torch.nn.Conv2d(3,3,k,1,'same',bias=False,padding_mode='reflect',groups=3)
        for param in self.blur_conv.parameters():
            param.requires_grad = False
        self.blur_conv.weight[:] = 1/(k**2)

        self.blur_conv = self.blur_conv.to(self.vqganDevice)
    else:
        self.blur_conv = None

    self.all_phrases = all_prompts'''