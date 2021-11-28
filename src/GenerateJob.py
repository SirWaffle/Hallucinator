import sys
import os
import random
import numpy as np


# shut off tqdm log spam by uncommenting the below
from tqdm import tqdm
import Hallucinator
# from functools import partialmethod
# tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

import makeCutouts
import imageUtils
import ImageMods


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
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))
    

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







class GenerationJob:
    # TODO: dont use argparse args, use a config / json / something
    # using argparseargs for now due to being in the middle of a refactor
    def __init__(self, hallucinatorInst: Hallucinator, argparseArgs ):
        ### this will define all classwide member variables, so its easy to see
        ## should really convert this to something that is more explicit, but that will come later
        self.config = argparseArgs
        self.hallucinatorInst = hallucinatorInst

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
    def AddImageMod(self, mod):   
        print('Initializing modifier: ' + str(mod))
        mod.Initialize()             
        self.ImageModifiers.append(mod)

    def OnPreTrain(self):
        pass


    ########
    ## new stuff im testing, very hackish here
    ########
    def InitSpatialPromptMasks(self):
        if self.config.use_spatial_prompts == False:
            return

        #Make prompt masks
        img = Image.open(self.config.prompt_key_image)
        pil_image = img.convert('RGB')
        #pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
        #pil_tensor = TF.to_tensor(pil_image)
        #self.quantizedImage, *_ = self.vqganModel.encode(pil_tensor.to(self.vqganDevice).unsqueeze(0) * 2 - 1)

        prompt_key_image = np.asarray(pil_image)

        #Set up color->prompt map
        color_to_prompt_idx={}
        all_prompts=[]
        for i,(color_key,blind,prompt) in enumerate(self.config.spatial_prompts):
            #append a collective promtp to all, to keep a set style if we want
            if prompt[-1]==' ':
                prompt+=self.config.append_to_prompts
            elif prompt[-1]=='.' or prompt[-1]=='|' or prompt[-1]==',':
                prompt+=" "+self.config.append_to_prompts
            else:
                prompt+=". "+self.config.append_to_prompts

            all_prompts.append(prompt)
            self.blindfold.append(blind)
            color_to_prompt_idx[color_key] = i
        
        color_to_prompt_idx_orig = dict(color_to_prompt_idx)

        #init the masks
        self.prompt_masks = torch.FloatTensor(
            len(self.config.spatial_prompts),
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
        if self.config.dilate_masks:
            struct_ele = torch.FloatTensor(1,1,self.config.dilate_masks,self.config.dilate_masks).fill_(1).to(self.vqganDevice)
            self.prompt_masks = F.conv2d(self.prompt_masks,struct_ele,padding='same')

        #resize masks to output size
        self.prompt_masks = F.interpolate(self.prompt_masks,(self.toksY * 16, self.toksX * 16))

        #make binary
        self.prompt_masks[self.prompt_masks>0.1]=1

        #rough display
        if self.prompt_masks.size(0)>=3:
            print('first 3 masks')
            TF.to_pil_image(self.prompt_masks[0,0].cpu()).save('ex-masks-0.png')   
            TF.to_pil_image(self.prompt_masks[1,0].cpu()).save('ex-masks-1.png')
            TF.to_pil_image(self.prompt_masks[2,0].cpu()).save('ex-masks-2.png')
            TF.to_pil_image(self.prompt_masks[0:3,0].cpu()).save('ex-masks-comb.png')
            #display.display(display.Image('ex-masks.png')) 
            if self.prompt_masks.size(0)>=6:
                print('next 3 masks')
                TF.to_pil_image(self.prompt_masks[3:6,0].cpu()).save('ex-masks.png')   
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
        self.InitPrompts()
        self.InitStartingImage()
        self.InitSpatialPromptMasks()

        self.CurrentCutoutMethod = self.hallucinatorInst.GetMakeCutouts( self.clipPerceptorInputResolution )
        
        # CLIP tokenize/encode
        if self.all_phrases and self.config.use_spatial_prompts:
            print("using masking images")
            for prompt in self.all_phrases:
                self.EmbedTextPrompt(prompt)
        elif self.config.prompts:
            print("using standard prompts")
            for prompt in self.config.prompts:
                self.EmbedTextPrompt(prompt)

        for prompt in self.config.image_prompts:
            path, weight, stop = split_prompt(prompt)
            img = Image.open(path)
            pil_image = img.convert('RGB')
            img = imageUtils.resize_image(pil_image, (self.ImageSizeX, self.ImageSizeY))
            batch = self.CurrentCutoutMethod(TF.to_tensor(img).unsqueeze(0).to(self.clipDevice))
            embed = self.clipPerceptor.encode_image(self.normalize(batch)).float()
            self.embededPrompts.append(Prompt(embed, weight, stop).to(self.clipDevice))

        for seed, weight in zip(self.config.noise_prompt_seeds, self.config.noise_prompt_weights):
            gen = torch.Generator().manual_seed(seed)
            embed = torch.empty([1, self.clipPerceptor.visual.output_dim]).normal_(generator=gen)
            self.embededPrompts.append(Prompt(embed, weight).to(self.clipDevice))

        self.optimiser = self.hallucinatorInst.get_optimiser(self.quantizedImage, self.config.optimiser,self.config.step_size)

        if self.config.optimiser == "MADGRAD":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimiser, 'min', factor=0.999, patience=0)  

        #for mod in self.ImageModifiers:
        #    print('Initializing modifier: ' + str(mod))
        #    mod.Initialize()


        # Output for the user
        print('Optimising using:', self.config.optimiser)

        if self.config.prompts:
            print('Using text prompts:', self.config.prompts)  
        if self.config.image_prompts:
            print('Using image prompts:', self.config.image_prompts)
        if self.config.init_image:
            print('Using initial image:', self.config.init_image)
        if self.config.noise_prompt_weights:
            print('Noise prompt weights:', self.config.noise_prompt_weights)    


    # this method should reset everything aside from CLIP & vqgan model loading
    # this should allow this to run in 'server mode' where it can get repeated requests
    # to generate without having unload / reload models
    def Reset(self):
        # TODO
        # can i just reset these variables and all is well? does stuff need to be detached and cleared out from CUDA devices?

        self.log_torch_mem('Pre PartialReset')

        self.quantizedImage = None
        self.optimiser = None     
        self.CurrentCutoutMethod = None
        self.embededPrompts = []
        self.all_phrases = []
        self.z_min = None
        self.z_max = None
        self.ImageSizeX = None
        self.ImageSizeY = None
        self.original_quantizedImage = None
        self.loss_idx = []
        self.scheduler = None

        gc.collect()
        torch.cuda.empty_cache()

        self.log_torch_mem('Post PartialReset')



    ##############
    ##  Getters and converters
    ##############
    def synth(self, z, gumbelMode):
        return self.hallucinatorInst.synth( self.quantizedImage, self.vqganGumbelEnabled)

    def GerCurrentImageAsPIL(self):
        out = self.hallucinatorInst.synth(self.quantizedImage, self.vqganGumbelEnabled)
        return TF.to_pil_image(out[0].cpu())

    def GetCurrentImageSynthed(self):
        return self.hallucinatorInst.synth( self.quantizedImage, self.vqganGumbelEnabled)

    def ConvertToPIL(self, synthedImage):
        return TF.to_pil_image(synthedImage[0].cpu())


    #####################
    ### Helper type methods
    #####################

    def EmbedTextPrompt(self, prompt):
        txt, weight, stop = split_prompt(prompt)
        embed = self.clipPerceptor.encode_text(clip.tokenize(txt).to(self.clipDevice)).float()
        self.embededPrompts.append(Prompt(embed, weight, stop).to(self.clipDevice))


    def InitPrompts(self):
        # Split text prompts using the pipe character (weights are split later)
        if self.config.prompts:
            # For stories, there will be many phrases
            story_phrases = [phrase.strip() for phrase in self.config.prompts.split("^")]
            
            # Make a list of all phrases
            for phrase in story_phrases:
                self.all_phrases.append(phrase.split("|"))
            
            # First phrase
            self.config.prompts = self.all_phrases[0]
            
        # Split target images using the pipe character (weights are split later)
        if self.config.image_prompts:
            self.config.image_prompts = self.config.image_prompts.split("|")
            self.config.image_prompts = [image.strip() for image in self.config.image_prompts]


    def InitStartingImage(self):
        vqganNumResolutionsF = 2**(self.vqganModel.decoder.num_resolutions - 1)
        self.toksX, self.toksY = self.config.size[0] // vqganNumResolutionsF, self.config.size[1] // vqganNumResolutionsF
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


        if self.config.init_image:
            if 'http' in self.config.init_image:
                img = Image.open(urlopen(self.config.init_image))
            else:
                img = Image.open(self.config.init_image)
                pil_image = img.convert('RGB')
                pil_image = pil_image.resize((self.ImageSizeX, self.ImageSizeY), Image.LANCZOS)
                pil_tensor = TF.to_tensor(pil_image)
                print( 'first encoding -> pil_tensor size: ' + str( pil_tensor.size() ) )
                self.quantizedImage, *_ = self.vqganModel.encode(pil_tensor.to(self.vqganDevice).unsqueeze(0) * 2 - 1)
        elif self.config.init_noise == 'pixels':
            img = imageUtils.random_noise_image(self.config.size[0], self.config.size[1])    
            pil_image = img.convert('RGB')
            pil_image = pil_image.resize((self.ImageSizeX, self.ImageSizeY), Image.LANCZOS)
            pil_tensor = TF.to_tensor(pil_image)
            self.quantizedImage, *_ = self.vqganModel.encode(pil_tensor.to(self.vqganDevice).unsqueeze(0) * 2 - 1)
        elif self.config.init_noise == 'gradient':
            img = imageUtils.random_gradient_image(self.config.size[0], self.config.size[1])
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


        if self.config.init_weight:
            self.original_quantizedImage = self.quantizedImage.detach() #was clone() and setting grad to false befopre

        self.quantizedImage.requires_grad_(True)