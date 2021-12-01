import sys
import os
import random
import numpy as np
from torch.functional import Tensor


# shut off tqdm log spam by uncommenting the below
from tqdm import tqdm
# from functools import partialmethod
# tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

import makeCutouts
import imageUtils
import GenerationMods
#import Hallucinator #circular reference in imports...

#stuff im using from source instead of installs
# i want to run clip from source, not an install. I have clip in a dir alongside this project
# so i append the parent dir to the proj and we expect to find a folder named clip there
sys.path.append('..\\')
from CLIP import clip


# pip install taming-transformers doesn't work with Gumbel, but does not yet work with coco etc
# appending the path does work with Gumbel
sys.path.append('taming-transformers')
from taming.models import cond_transformer, vqgan



import yaml
from urllib.request import urlopen
import gc

from omegaconf import OmegaConf

import torch
from torch.cuda.amp import autocast
from torch.cuda.amp import custom_fwd
from torch.cuda.amp import custom_bwd
from torch.cuda.amp import GradScaler
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.cuda import get_device_properties

import torch_optimizer

import imageio

from PIL import ImageFile, Image, PngImagePlugin, ImageChops
ImageFile.LOAD_TRUNCATED_IMAGES = True

from subprocess import Popen, PIPE
import re

from torchvision.datasets import CIFAR100
 



def build_filename_path( outputDir, filename ):
    fullpath = outputDir
    if fullpath[-1] != '\\' and fullpath[-1] != '/':
        fullpath += '/'
    fullpath += filename
    return fullpath


#########################
### misc functions that need to be sorted properly
#########################
class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)

replace_grad = ReplaceGrad.apply


class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf'), textPrompt = None):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))     
        self.TextPrompt = textPrompt
        if self.TextPrompt == None:
            self.TextPrompt = 'not a text prompt'
    

    @autocast(enabled=makeCutouts.use_mixed_precision)
    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()


#NR: Split prompts and weights
def split_prompt(prompt):
    vals = prompt.rsplit(':', 2)
    vals = vals + ['', '1', '-inf'][len(vals):]
    return vals[0], float(vals[1]), float(vals[2])



## make this something we can init in code or whatever, for nwo this so i can clear argparse shit from these classes
class SpatialPromptConfig:
    def __init__(self):
        self.spatial_prompts=[
            ( (255,0,0), 0.1, '''teeth beksinski'''),
            ( (0,255,0), 0.1, '''demon gustave dore'''),
            ( (0,0,255), 0.1, '''eggs giger'''),
            ( (0,0,0), 0.1, '''stained glass'''),
        ]
        
        self.append_to_prompts = ''
        self.prompt_key_image = './examples/4-color-mask.png'
        self.dilate_masks = 10
        #self.use_spatial_prompts=True
        #### /end hacky testing for spatial prompts
        self.use_spatial_prompts = True




class GenerationJob:
    # TODO: dont use argparse args, use a config / json / something
    # using argparseargs for now due to being in the middle of a refactor
    def __init__(self, hallucinatorInst, cut_method:str = "latest", totalIterations:int = 200, prompts:str = "A waffle and a squishbrain", image_prompts = [], 
                 spatialPromptConfig:SpatialPromptConfig = None, startingImage:str = None, imageSizeXY:tuple = [512, 512], 
                 cutNum:int = 32, cutSize:tuple = [0,0], cutPow:float = 1.0, augments:list = [], optimiserName:str = "Adam", stepSize:float = 0.1,
                 init_weight:float = 0., init_noise:str = "random", noise_prompt_seeds = [], noise_prompt_weights=[], prompt_frequency:int = 0,
                 deterministic:int = 0, outputDir:str = './output/', outputFilename:str = 'output.png', save_freq:int = 50, save_seq:bool = False, 
                 save_best:bool = False, useKorniaAugmentsInsteadOfTorchTransforms:bool = True):

        ## config variables passed in
        self.savedImageCount = 0    #used to track how many images we saved
        self.bestErrorScore = 99999 # used to track the best error score we have seen
        self.outputFilename = outputFilename
        self.outputDir = outputDir
        self.save_freq = save_freq
        self.save_seq = save_seq
        self.save_best = save_best

        self.cut_method = cut_method
        self.totalIterations = totalIterations
        self.prompts = prompts
        self.image_prompts = image_prompts
        self.init_image = startingImage
        self.size = imageSizeXY
        self.cutn = cutNum
        self.cut_size = cutSize
        self.cut_pow = cutPow
        self.augments = augments
        self.optimiserName = optimiserName
        self.step_size = stepSize
        self.init_weight = init_weight
        self.init_noise = init_noise
        self.noise_prompt_seeds = noise_prompt_seeds
        self.noise_prompt_weights = noise_prompt_weights
        self.prompt_frequency = prompt_frequency
        self.deterministic = deterministic
        self.korniaAugments = useKorniaAugmentsInsteadOfTorchTransforms

        self.spatialPromptConfig = spatialPromptConfig

        self.use_spatial_prompts = False
        if spatialPromptConfig != None and spatialPromptConfig.use_spatial_prompts == True:
            self.use_spatial_prompts = True


        self.hallucinatorInst = hallucinatorInst
        self.use_mixed_precision = self.hallucinatorInst.use_mixed_precision

        ## public vars, is there a better way to do this in python?        
        self.currentIteration = 0 
        self.phraseCounter = 0 #phrase counter stuff from old project...


        ## maybe not best for logn run, but get ref's to some things that are heavily used from hallucinator
        self.clipPerceptorInputResolution = hallucinatorInst.clipPerceptorInputResolution
        self.clipPerceptor = hallucinatorInst.clipPerceptor
        self.clipDevice = hallucinatorInst.clipDevice

        self.vqganDevice = hallucinatorInst.vqganDevice
        self.vqganModel = hallucinatorInst.vqganModel
        self.vqganGumbelEnabled = hallucinatorInst.vqganGumbelEnabled



        self.quantizedImage = None # source image thats fed into taming transformers

        self.optimiser = None #currently in use optimiser        

        # cuts
        self.CurrentCutoutMethod = None

        # prompts
        self.embededPrompts = []
        self.all_phrases = []

        #### these need better names, wtf are they exactly?
        self.z_min = None
        self.z_max = None
        self.toksY = None
        self.toksX = None
        self.ImageSizeX = None
        self.ImageSizeY = None
        self.original_quantizedImage = None

        #MADGRAD related, needs better naming
        self.loss_idx = []
        self.scheduler = None

        #mixed precision scaler
        self.gradScaler = GradScaler()
        
        # From imagenet - Which is better?
        #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                        std=[0.26862954, 0.26130258, 0.27577711])



        # masking stuff
        self.num_prompts = 0
        self.blur_conv = None
        self.prompt_masks = None
        self.blindfold = []
        self.noise_fac = 0.1


        #image modifications
        self.ImageModifiers = []



    ####################
    ### Image modifier stuff
    ####################

    # method that uses the previous mod's end frame to set the start frame
    def AppendGenerationMod(self, mod:GenerationMods.IGenerationMod, numberOfIts: int, freq: int) -> int: 
        startFrame = 0
        modLen = len(self.ImageModifiers)
        if modLen > 0:
            startFrame = self.ImageModifiers[modLen -1].endIt + 1
        return self.AddGenerationMod(mod, startFrame, numberOfIts, freq)

    # add a new mod, and return the next frame after it
    def AddGenerationMod(self, mod:GenerationMods.IGenerationMod, startIt: int, numberOfIts: int, freq: int) -> int:   
        print('Initializing modifier: ' + str(mod))
        mod.Initialize()   

        endIt = startIt + numberOfIts
        modContainer = GenerationMods.GenerationModContainer(mod, startIt, endIt, freq)
        self.ImageModifiers.append(modContainer)
        return endIt + 1

    def AddGenerationModOneShot(self, mod:GenerationMods.IGenerationMod, startIt: int) -> int:
        return self.AddGenerationMod(mod, startIt, 0, 1)

    def OnPreTrain(self):
        pass


    ##############
    ##  Getters and converters
    ##############
    def GetCurrentImageAsPIL(self) -> torch.Tensor:
        return self.ConvertToPIL( self.GetCurrentImageSynthed() )

    def GetCurrentImageSynthed(self):
        return self.hallucinatorInst.synth( self.quantizedImage, self.vqganGumbelEnabled)

    def ConvertToPIL(self, synthedImage):
        return TF.to_pil_image(synthedImage[0].detach().cpu())


    ######
    # save file functions
    #####
    def SaveImageTensor( self, imgTensor:torch.Tensor, filenamePrefix:str = None, info:PngImagePlugin.PngInfo = None ):
        self.SaveImage( self.ConvertToPIL(imgTensor), filenamePrefix, info)

    def SaveCurrentImage( self, filenamePrefix:str = None, info:PngImagePlugin.PngInfo = None ):
        pilImage = self.GetCurrentImageAsPIL()
        self.SaveImage( pilImage, filenamePrefix, info)

    def SaveImage( self, pilImage, filenamePrefix:str = None, info:PngImagePlugin.PngInfo = None ):
        if filenamePrefix != None:
            outName = filenamePrefix + self.outputFilename
        else:
            outName = self.outputFilename

        
        if info == None:
            pilImage.save( build_filename_path( self.outputDir, outName ))
        else:
            pilImage.save( build_filename_path( self.outputDir, outName ), pnginfo=info)

        del pilImage

        

    ########
    ## new stuff im testing, very hackish here
    ########
    def InitSpatialPromptMasks(self):
        if self.use_spatial_prompts == False:
            return

        #Make prompt masks
        img = Image.open(self.spatialPromptConfig.prompt_key_image)
        pil_image = img.convert('RGB')
        #pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
        #pil_tensor = TF.to_tensor(pil_image)
        #self.quantizedImage, *_ = self.vqganModel.encode(pil_tensor.to(self.vqganDevice).unsqueeze(0) * 2 - 1)

        prompt_key_image = np.asarray(pil_image)

        #Set up color->prompt map
        color_to_prompt_idx={}
        all_prompts=[]
        for i,(color_key,blind,prompt) in enumerate(self.spatialPromptConfig.spatial_prompts):
            #append a collective promtp to all, to keep a set style if we want
            if prompt[-1]==' ':
                prompt+=self.spatialPromptConfig.append_to_prompts
            elif prompt[-1]=='.' or prompt[-1]=='|' or prompt[-1]==',':
                prompt+=" "+self.spatialPromptConfig.append_to_prompts
            else:
                prompt+=". "+self.spatialPromptConfig.append_to_prompts

            all_prompts.append(prompt)
            self.blindfold.append(blind)
            color_to_prompt_idx[color_key] = i
        
        color_to_prompt_idx_orig = dict(color_to_prompt_idx)

        #init the masks
        self.prompt_masks = torch.FloatTensor(
            len(self.spatialPromptConfig.spatial_prompts),
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

                self.prompt_masks[idx,0,y,x]=1

        self.prompt_masks = self.prompt_masks.to(self.vqganDevice)

        #dilate masks to prevent possible disontinuity artifacts
        if self.spatialPromptConfig.dilate_masks:
            struct_ele = torch.FloatTensor(1,1,self.spatialPromptConfig.dilate_masks,self.spatialPromptConfig.dilate_masks).fill_(1).to(self.vqganDevice)
            self.prompt_masks = F.conv2d(self.prompt_masks,struct_ele,padding='same')

        #resize masks to output size
        self.prompt_masks = F.interpolate(self.prompt_masks,(self.toksY * 16, self.toksX * 16))

        #make binary
        self.prompt_masks[self.prompt_masks>0.1]=1

        #rough display
        if self.prompt_masks.size(0)>=3:
            print('first 3 masks')
            TF.to_pil_image(self.prompt_masks[0,0].detach().cpu()).save('ex-masks-0.png')   
            TF.to_pil_image(self.prompt_masks[1,0].detach().cpu()).save('ex-masks-1.png')
            TF.to_pil_image(self.prompt_masks[2,0].detach().cpu()).save('ex-masks-2.png')
            TF.to_pil_image(self.prompt_masks[0:3,0].detach().cpu()).save('ex-masks-comb.png')
            #display.display(display.Image('ex-masks.png')) 
            if self.prompt_masks.size(0)>=6:
                print('next 3 masks')
                TF.to_pil_image(self.prompt_masks[3:6,0].detach().cpu()).save('ex-masks.png') 
                #display.display(display.Image('ex-masks.png')) 
        
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

        self.all_phrases = all_prompts
        self.num_prompts = len(all_prompts)



    #############
    ## Life cycle
    #############

    # stuff that needs to be initialized per request / reset between requests
    def Initialize(self):

        print('creating directory: ' + os.path.dirname(self.outputDir))
        os.makedirs(os.path.dirname(self.outputDir), exist_ok=True)

        if self.deterministic >= 2:
            print("GenerationJob Determinism at max: forcing a lot of things so this will work, no augs, non-pooling cut method, bad resampling")
            self.augments = "none"
            self.cut_method = "original"
            # need to make cutouts use deterministic stuff... probably not a good way
            makeCutouts.deterministic = True
        elif self.deterministic == 1:
            print("GenerationJob Determinism at medium: no changes needed")
        else:
            print("GenerationJob Determinism at minimum: no changes needed")

        if not self.augments:
            self.augments = [['Af', 'Pe', 'Ji', 'Er']]
        elif self.augments == 'None':
            print("Augments set to none")
            self.augments = []

        if self.use_mixed_precision==True:
            print("GenerationJob: cant use augments in mixed precision mode yet...")
            self.augments = [] 


        self.InitPrompts()
        self.InitStartingImage()
        self.InitSpatialPromptMasks()

        self.CurrentCutoutMethod = makeCutouts.GetMakeCutouts( self.cut_method, self.clipPerceptorInputResolution, self.cutn, self.cut_size, self.cut_pow, self.augments, self.korniaAugments )
        
        # CLIP tokenize/encode
        if self.all_phrases and self.use_spatial_prompts:
            print("using masking images")
            for prompt in self.all_phrases:
                self.EmbedTextPrompt(prompt)
        elif self.prompts:
            print("using standard prompts")
            for prompt in self.prompts:
                self.EmbedTextPrompt(prompt)

        for prompt in self.image_prompts:
            path, weight, stop = split_prompt(prompt)
            img = Image.open(path)
            pil_image = img.convert('RGB')
            img = imageUtils.resize_image(pil_image, (self.ImageSizeX, self.ImageSizeY))
            batch = self.CurrentCutoutMethod(TF.to_tensor(img).unsqueeze(0).to(self.clipDevice))
            embed = self.clipPerceptor.encode_image(self.normalize(batch)).float()
            self.embededPrompts.append(Prompt(embed, weight, stop).to(self.clipDevice))

        for seed, weight in zip(self.noise_prompt_seeds, self.noise_prompt_weights):
            gen = torch.Generator().manual_seed(seed)
            embed = torch.empty([1, self.clipPerceptor.visual.output_dim]).normal_(generator=gen)
            self.embededPrompts.append(Prompt(embed, weight).to(self.clipDevice))

        self.optimiser = self.hallucinatorInst.get_optimiser(self.quantizedImage, self.optimiserName, self.step_size)

        if self.optimiser == "MADGRAD":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimiser, 'min', factor=0.999, patience=0)  

        #for mod in self.ImageModifiers:
        #    print('Initializing modifier: ' + str(mod))
        #    mod.Initialize()


        # Output for the user
        print('Optimising using:', self.optimiser)

        if self.prompts:
            print('Using text prompts:', self.prompts)  
        if self.image_prompts:
            print('Using image prompts:', self.image_prompts)
        if self.init_image:
            print('Using initial image:', self.init_image)
        if self.noise_prompt_weights:
            print('Noise prompt weights:', self.noise_prompt_weights)    



    #####################
    ### Helper type methods
    #####################

    def EmbedTextPrompt(self, prompt):
        txt, weight, stop = split_prompt(prompt)
        embed = self.clipPerceptor.encode_text(clip.tokenize(txt).to(self.clipDevice)).float()
        self.embededPrompts.append(Prompt(embed, weight, stop, txt).to(self.clipDevice))


    def InitPrompts(self):
        # Split text prompts using the pipe character (weights are split later)
        if self.prompts:
            # For stories, there will be many phrases
            story_phrases = [phrase.strip() for phrase in self.prompts.split("^")]
            
            # Make a list of all phrases
            for phrase in story_phrases:
                self.all_phrases.append(phrase.split("|"))
            
            # First phrase
            self.prompts = self.all_phrases[0]
            
        # Split target images using the pipe character (weights are split later)
        if self.image_prompts:
            self.image_prompts = self.image_prompts.split("|")
            self.image_prompts = [image.strip() for image in self.image_prompts]


    def InitStartingImage(self):
        vqganNumResolutionsF = 2**(self.vqganModel.decoder.num_resolutions - 1)
        self.toksX, self.toksY = self.size[0] // vqganNumResolutionsF, self.size[1] // vqganNumResolutionsF
        self.ImageSizeX, self.ImageSizeY = self.toksX * vqganNumResolutionsF, self.toksY * vqganNumResolutionsF

        print("vqgan input resolutions: " + str(self.vqganModel.decoder.num_resolutions))
        print("cliperceptor input_res (aka cut size): " + str(self.clipPerceptorInputResolution) + " and whatever f is supposed to be: " + str(vqganNumResolutionsF))
        print("Toks X,Y: " + str(self.toksX) + ", " + str(self.toksY) + "      SizeX,Y: " + str(self.ImageSizeX) + ", " + str(self.ImageSizeY))
        
        # Gumbel or not?
        if self.vqganGumbelEnabled:
            e_dim = 256
            n_toks = self.vqganModel.quantize.n_embed
            self.z_min = self.vqganModel.quantize.embed.weight.min(dim=0).values[None, :, None, None]
            self.z_max = self.vqganModel.quantize.embed.weight.max(dim=0).values[None, :, None, None]
        else:
            e_dim = self.vqganModel.quantize.e_dim
            n_toks = self.vqganModel.quantize.n_e
            self.z_min = self.vqganModel.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
            self.z_max = self.vqganModel.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

        print('initializing image of size: ' + str(self.size))

        if self.init_image:
            if 'http' in self.init_image:
                img = Image.open(urlopen(self.init_image))
            else:
                img = Image.open(self.init_image)
                pil_image = img.convert('RGB')
                pil_image = pil_image.resize((self.ImageSizeX, self.ImageSizeY), Image.LANCZOS)
                pil_tensor = TF.to_tensor(pil_image)
                print( 'first encoding -> pil_tensor size: ' + str( pil_tensor.size() ) )
                self.quantizedImage, *_ = self.vqganModel.encode(pil_tensor.to(self.vqganDevice).unsqueeze(0) * 2 - 1)
        elif self.init_noise == 'pixels':
            img = imageUtils.random_noise_image(self.size[0], self.size[1])    
            pil_image = img.convert('RGB')
            pil_image = pil_image.resize((self.ImageSizeX, self.ImageSizeY), Image.LANCZOS)
            pil_tensor = TF.to_tensor(pil_image)
            self.quantizedImage, *_ = self.vqganModel.encode(pil_tensor.to(self.vqganDevice).unsqueeze(0) * 2 - 1)
        elif self.init_noise == 'gradient':
            img = imageUtils.random_gradient_image(self.size[0], self.size[1])
            pil_image = img.convert('RGB')
            pil_image = pil_image.resize((self.ImageSizeX, self.ImageSizeY), Image.LANCZOS)
            pil_tensor = TF.to_tensor(pil_image)
            self.quantizedImage, *_ = self.vqganModel.encode(pil_tensor.to(self.vqganDevice).unsqueeze(0) * 2 - 1)
        else:
            one_hot = F.one_hot(torch.randint(n_toks, [self.toksY * self.toksX], device=self.vqganDevice), n_toks).float()
            # z = one_hot @ vqganModel.quantize.embedding.weight
            if self.vqganGumbelEnabled:
                self.quantizedImage = one_hot @ self.vqganModel.quantize.embed.weight
            else:
                self.quantizedImage = one_hot @ self.vqganModel.quantize.embedding.weight

            self.quantizedImage = self.quantizedImage.view([-1, self.toksY, self.toksX, e_dim]).permute(0, 3, 1, 2) 
            #z = torch.rand_like(z)*2						# NR: check


        if self.init_weight:
            self.original_quantizedImage = self.quantizedImage.detach() #was clone() and setting grad to false befopre

        self.quantizedImage.requires_grad_(True)



    def GetCutouts(self, synthedImage):
        cutouts, cutout_coords = self.CurrentCutoutMethod(synthedImage)

        # attempt masking stuff
        if self.use_spatial_prompts:
            cutouts_detached = cutouts.detach() #used to prevent gradient for unmask parts
            if self.blur_conv is not None:
                #Get the "blindfolded" image by blurring then addimg more noise
                facs = cutouts.new_empty([cutouts.size(0), 1, 1, 1]).uniform_(0, self.noise_fac)
                cutouts_blurred = self.blur_conv(cutouts_detached)+ facs * torch.randn_like(cutouts_detached)


            cut_size = self.cut_size

            #get mask patches
            cutout_prompt_masks = []
            for (x1,x2,y1,y2) in cutout_coords:
                cutout_mask = self.prompt_masks[:,:,y1:y2,x1:x2]
                cutout_mask = makeCutouts.resample(cutout_mask, (cut_size[0], cut_size[1]))
                cutout_prompt_masks.append(cutout_mask)
            cutout_prompt_masks = torch.stack(cutout_prompt_masks,dim=1) #-> prompts X cutouts X color X H X W
            
            #apply each prompt, masking gradients
            prompts_gradient_masked_cutouts = []
            for idx,prompt in enumerate(self.embededPrompts):
                keep_mask = cutout_prompt_masks[idx] #-> cutouts X color X H X W
                #only apply this prompt if one image has a (big enough) part of mask
                if keep_mask.sum(dim=3).sum(dim=2).max()> cut_size[0]*2: #todo, change this
                    
                    block_mask = 1-keep_mask

                    #compose cutout of gradient and non-gradient parts
                    if self.blindfold[idx] and ((not isinstance(self.blindfold[idx],float)) or self.blindfold[idx]>random.random()):
                        gradient_masked_cutouts = keep_mask*cutouts + block_mask*cutouts_blurred
                    else:
                        gradient_masked_cutouts = keep_mask*cutouts + block_mask*cutouts_detached

                    prompts_gradient_masked_cutouts.append(gradient_masked_cutouts)
            cutouts = torch.cat(prompts_gradient_masked_cutouts,dim=0)    

        return cutouts  


    def GetCutoutResults(self, clipEncodedImage, iteration:int):
        result = [] 

        if self.init_weight:
            # result.append(F.mse_loss(z, z_orig) * args.init_weight / 2)
            result.append(F.mse_loss(self.quantizedImage, torch.zeros_like(self.original_quantizedImage)) * ((1/torch.tensor(iteration*2 + 1))*self.init_weight) / 2)

        if self.use_spatial_prompts:
            for prompt_masked_iii,prompt in zip(torch.chunk(clipEncodedImage,self.num_prompts,dim=0),self.embededPrompts):
                result.append(prompt(prompt_masked_iii))
        else:
            for prompt in self.embededPrompts:
                result.append(prompt(clipEncodedImage))      

        return result