import sys
import os
import random
from typing import List
import numpy as np
from torch.functional import Tensor

from src import MakeCutouts
from src import ImageUtils
from src import GenerationCommands
from src import GenerationCommand

#import Hallucinator #circular reference in imports...

#stuff im using from source instead of installs
# i want to run clip from source, not an install. I have clip in a dir alongside this project
# so i append the parent dir to the proj and we expect to find a folder named clip there
sys.path.append('..\\')
from CLIP import clip
#from clip import clip


# pip install taming-transformers doesn't work with Gumbel, but does not yet work with coco etc
# appending the path does work with Gumbel
sys.path.append('taming-transformers')
from taming.models import cond_transformer, vqgan



from urllib.request import urlopen

import torch
from torch.cuda.amp import autocast
from torch.cuda.amp import custom_fwd
from torch.cuda.amp import custom_bwd
from torch.cuda.amp import GradScaler
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
import torch_optimizer

from PIL import ImageFile, Image, PngImagePlugin, ImageChops
ImageFile.LOAD_TRUNCATED_IMAGES = True
 


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
    def __init__(self, embed, weight=1., stop=float('-inf'), textPrompt:str = None, promptMask:torch.Tensor = None, maskBlindfold:float = None):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop)) 

        self.TextPrompt = textPrompt
        if self.TextPrompt == None:
            self.TextPrompt = 'not a text prompt'

        self.promptMask = promptMask   # mask associated with this prompt
        self.maskBlindfold = maskBlindfold #blindfold... TODO: dewscribe better when if igure out what it does exactly
    

    @autocast(enabled=MakeCutouts.use_mixed_precision)
    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()


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




class GenerationJob:
    def __init__(self, hallucinatorInst, totalIterations:int = 200, image_prompts = [], 
                 startingImage:str = None, imageSizeXY:List[int] = [512, 512], 
                 init_weight:float = 0., init_noise:str = "random", noise_prompt_seeds = [], noise_prompt_weights=[],
                 deterministic:int = 0, outputDir:str = './output/', outputFilename:str = 'output.png', save_freq:int = 50, save_seq:bool = False, 
                 save_best:bool = False):

        ## config variables passed in
        self.savedImageCount = 0    #used to track how many images we saved
        self.bestErrorScore = 99999 # used to track the best error score we have seen
        self.outputFilename = outputFilename
        self.outputDir = outputDir
        self.save_freq = save_freq
        self.save_seq = save_seq
        self.save_best = save_best

        self.totalIterations = totalIterations
        self.image_prompts = image_prompts
        self.init_image = startingImage
        self.ImageSizeXY = imageSizeXY
        self.init_weight = init_weight
        self.init_noise = init_noise
        self.noise_prompt_seeds = noise_prompt_seeds
        self.noise_prompt_weights = noise_prompt_weights
        self.deterministic = deterministic

        
        self.hallucinatorInst = hallucinatorInst

        ## public vars, is there a better way to do this in python?        
        self.currentIteration = 0 


        ## maybe not best for logn run, but get ref's to some things that are heavily used from hallucinator
        self.clipPerceptorInputResolution = hallucinatorInst.clipPerceptorInputResolution
        self.clipPerceptor = hallucinatorInst.clipPerceptor
        self.clipDevice = hallucinatorInst.clipDevice

        self.vqganDevice = hallucinatorInst.vqganDevice
        self.vqganModel = hallucinatorInst.vqganModel
        self.vqganGumbelEnabled = hallucinatorInst.vqganGumbelEnabled



        self.quantizedImage: torch.Tensor = None # source image thats fed into taming transformers

        self.optimizer: torch.optim.Optimizer = None #currently in use optimizer    
        self.optimizerName = "Adam"
        self.optimizerLearningRate = 0.1    

        # cuts
        self.CurrentCutoutMethod = None

        # prompts
        self.embededPrompts: List[Prompt] = []

        #### these need better names, wtf are they exactly?
        self.z_min = None
        self.z_max = None
        self.ImageSizeX:int = None
        self.ImageSizeY:int = None
        self.original_quantizedImage:torch.Tensor = None

        #MADGRAD related, needs better naming
        self.loss_idx: List[float] = []
        self.scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau = None

        #mixed precision scaler
        self.gradScaler = GradScaler()
        
        # From imagenet - Which is better?
        #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                        std=[0.26862954, 0.26130258, 0.27577711])



        # masking/spatial prompt stuff
        self.use_spatial_prompts = False
        self.blur_conv = None   # this is set from the job 
        self.noise_fac = 0.1     # used by the blur function


        #image modifications
        self.GenerationCommandList: List[GenerationCommand.GenerationCommandContainer] = []



    ####################
    ### Commands
    ####################

    # method that uses the last command in the lists end frame to set the start frame
    def AppendGenerationCommand(self, mod:GenerationCommand.IGenerationCommand, numberOfIts: int, freq: int) -> int: 
        startFrame = 0
        modLen = len(self.GenerationCommandList)
        if modLen > 0:
            startFrame = self.GenerationCommandList[modLen -1].endIt + 1
        return self.AddGenerationCommand(mod, startFrame, numberOfIts, freq)

    # add a new mod, and return the next frame after it
    def AddGenerationCommand(self, mod:GenerationCommand.IGenerationCommand, startIt: int, numberOfIts: int, freq: int) -> int:   
        print('----> Add and init command: start: ' + str( startIt ) + ', duration: ' + str(numberOfIts) + ', frequency: ' + str(freq) + ', command: ' + str(type(mod)))
        mod.Initialize()   

        endIt = startIt + numberOfIts
        modContainer = GenerationCommand.GenerationCommandContainer(mod, startIt, endIt, freq)
        self.GenerationCommandList.append(modContainer)
        return endIt + 1

    def AddGenerationCommandFireOnce(self, mod:GenerationCommand.IGenerationCommand, iterationToExecute: int) -> int:
        return self.AddGenerationCommand(mod, iterationToExecute, 0, 1)

    ##############################
    ## Modifier events
    ##############################
    def OnPreTrain(self):
        for modContainer in self.GenerationCommandList:
            if modContainer.ShouldApply( GenerationCommand.GenerationModStage.PreTrain, self.currentIteration ):
                modContainer.OnExecute( self.currentIteration )

        # check to ensure we important things set, otherwise default them
        if self.currentIteration == 0:
            print("PRETRAIN FIRST CALL")
            # creates a default cut method. this is expected to be set by the SetCutMethod command
            if self.CurrentCutoutMethod == None:            
                self.SetCutMethod()
                print("created default cutmethod: " + str(type(self.CurrentCutoutMethod)))

            # create the default optimizer if none specified
            if self.optimizer == None:
                self.SetOptimizer()
                print("created default optimiser: " + str(self.optimizer))


    def OnFinishGeneration(self):
        for modContainer in self.GenerationCommandList:
            if modContainer.ShouldApply( GenerationCommand.GenerationModStage.FinishedGeneration, self.currentIteration ):
                modContainer.OnExecute( self.currentIteration )

    ##############
    ##  image Getters and converters
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

    ##########################
    ## Prompt embedding
    ##########################
    
    #NR: Split prompts and weights
    def split_prompt(self, prompt):
        vals = prompt.rsplit(':', 2)
        vals = vals + ['', '1', '-inf'][len(vals):]
        return vals[0], float(vals[1]), float(vals[2])

    def EmbedTextPrompt(self, prompt:str):
        txt, weight, stop = self.split_prompt(prompt)
        embed = self.clipPerceptor.encode_text(clip.tokenize(txt).to(self.clipDevice)).float()
        self.embededPrompts.append(Prompt(embed, weight, stop, txt).to(self.clipDevice))


    def EmbedMaskedPrompt(self, prompt:str, promptMask:torch.Tensor = None, blindfold:float = 0.1):
        txt, weight, stop = self.split_prompt(prompt)
        embed = self.clipPerceptor.encode_text(clip.tokenize(txt).to(self.clipDevice)).float()
        self.embededPrompts.append(Prompt(embed, weight, stop, txt, promptMask, blindfold).to(self.clipDevice))

    ###################
    ##  some getters and setters...
    ###################

    def SetCutMethod(self, cutMethod:str = 'latest', cutNum:int = 32, cutSize:List[int] = [0, 0], cutPow:float = 1.0, augmentNameList:list = [], use_kornia:bool = True ):
        if not augmentNameList:
            print("adding default augments, since none were provided")
            augmentNameList = [['Af', 'Pe', 'Ji', 'Er']]
        elif augmentNameList == 'None':
            print("Augments set to none")
            augmentNameList = []

        if self.deterministic >= 2:
            print("GenerationJob Determinism at max: forcing a lot of things so this will work, no augs, non-pooling cut method, bad resampling")
            augmentNameList = []
            cutMethod = "original"
            # need to make cutouts use deterministic stuff... probably not a good way
            MakeCutouts.deterministic = True
        elif self.deterministic == 1:
            print("GenerationJob Determinism at medium: no changes needed")
        else:
            print("GenerationJob Determinism at minimum: no changes needed")

        if self.hallucinatorInst.use_mixed_precision == True:
            print("GenerationJob: cant use augments in mixed precision mode yet...")
            augmentNameList = [] 

        self.CurrentCutoutMethod = MakeCutouts.GetMakeCutouts( cutMethod, self.clipPerceptorInputResolution, cutNum, cutSize, cutPow, augmentNameList, use_kornia )


    ########################
    # get the optimizer ###
    ########################

    def get_optimizer(self, quantizedImg:torch.Tensor, opt_name:str, opt_lr:float):

        # from nerdy project, potential learning rate tweaks?
        # Messing with learning rate / optimizers
        #variable_lr = args.step_size
        #optimizer_list = [['Adam',0.075],['AdamW',0.125],['Adagrad',0.2],['Adamax',0.125],['DiffGrad',0.075],['RAdam',0.125],['RMSprop',0.02]]

        opt: torch.optim.Optimizer = None
        if opt_name == "Adam":
            opt = optim.Adam([quantizedImg], lr=opt_lr)	# LR=0.1 (Default)
        elif opt_name == "AdamW":
            opt = optim.AdamW([quantizedImg], lr=opt_lr)	
        elif opt_name == "Adagrad":
            opt = optim.Adagrad([quantizedImg], lr=opt_lr)	
        elif opt_name == "Adamax":
            opt = optim.Adamax([quantizedImg], lr=opt_lr)	
        elif opt_name == "DiffGrad":
            opt = torch_optimizer.DiffGrad([quantizedImg], lr=opt_lr, eps=1e-9, weight_decay=1e-9) # NR: Playing for reasons
        elif opt_name == "AdamP":
            opt = torch_optimizer.AdamP([quantizedImg], lr=opt_lr)		    	    
        elif opt_name == "RMSprop":
            opt = optim.RMSprop([quantizedImg], lr=opt_lr)
        elif opt_name == "MADGRAD":
            opt = torch_optimizer.MADGRAD([quantizedImg], lr=opt_lr)      
        else:
            print("Unknown optimizer. Are choices broken?")
            opt = optim.Adam([quantizedImg], lr=opt_lr)
        return opt


    def SetOptimizer(self, opt_name:str = "Adam", opt_lr:float = 0.1) -> None:
        self.optimizer = self.get_optimizer(self.quantizedImage, opt_name, opt_lr)

        if self.optimizer == "MADGRAD":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.999, patience=0)  

        self.optimizerName = opt_name
        self.optimizerLearningRate = opt_lr 

        # Output for the user
        print('Optimizing using:', self.optimizer)        


    #############
    ## Life cycle
    #############

    # stuff that needs to be initialized per request / reset between requests
    def Initialize(self):

        print('creating directory: ' + os.path.dirname(self.outputDir))
        os.makedirs(os.path.dirname(self.outputDir), exist_ok=True)

        self.InitPrompts()
        self.InitStartingImage()
        
        # CLIP tokenize/encode
        for prompt in self.image_prompts:
            path, weight, stop = self.split_prompt(prompt)
            img = Image.open(path)
            pil_image = img.convert('RGB')
            img = ImageUtils.resize_image(pil_image, (self.ImageSizeX, self.ImageSizeY))
            batch = self.CurrentCutoutMethod(TF.to_tensor(img).unsqueeze(0).to(self.clipDevice))
            embed = self.clipPerceptor.encode_image(self.normalize(batch)).float()
            self.embededPrompts.append(Prompt(embed, weight, stop).to(self.clipDevice))

        for seed, weight in zip(self.noise_prompt_seeds, self.noise_prompt_weights):
            gen = torch.Generator().manual_seed(seed)
            embed = torch.empty([1, self.clipPerceptor.visual.output_dim]).normal_(generator=gen)
            self.embededPrompts.append(Prompt(embed, weight).to(self.clipDevice))

        if self.image_prompts:
            print('Using image prompts:', self.image_prompts)
        if self.init_image:
            print('Using initial image:', self.init_image)
        if self.noise_prompt_weights:
            print('Noise prompt weights:', self.noise_prompt_weights)    


    #####################
    ### Helper type methods
    #####################

    def InitPrompts(self):
           
        # Split target images using the pipe character (weights are split later)
        if self.image_prompts:
            self.image_prompts = self.image_prompts.split("|")
            self.image_prompts = [image.strip() for image in self.image_prompts]


    def InitStartingImage(self):
        vqganNumResolutionsF = 2**(self.vqganModel.decoder.num_resolutions - 1)
        toksX, toksY = self.ImageSizeXY[0] // vqganNumResolutionsF, self.ImageSizeXY[1] // vqganNumResolutionsF
        self.ImageSizeX, self.ImageSizeY = toksX * vqganNumResolutionsF, toksY * vqganNumResolutionsF

        print("vqgan input resolutions: " + str(self.vqganModel.decoder.num_resolutions))
        print("cliperceptor input_res (aka cut size): " + str(self.clipPerceptorInputResolution) + " and whatever f is supposed to be: " + str(vqganNumResolutionsF))
        print("Toks X,Y: " + str(toksX) + ", " + str(toksY) + "      SizeX,Y: " + str(self.ImageSizeX) + ", " + str(self.ImageSizeY))
        
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

        print('initializing image of size: ' + str(self.ImageSizeXY))

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
            img = ImageUtils.random_noise_image(self.ImageSizeXY[0], self.ImageSizeXY[1])    
            pil_image = img.convert('RGB')
            pil_image = pil_image.resize((self.ImageSizeX, self.ImageSizeY), Image.LANCZOS)
            pil_tensor = TF.to_tensor(pil_image)
            self.quantizedImage, *_ = self.vqganModel.encode(pil_tensor.to(self.vqganDevice).unsqueeze(0) * 2 - 1)
        elif self.init_noise == 'gradient':
            img = ImageUtils.random_gradient_image(self.ImageSizeXY[0], self.ImageSizeXY[1])
            pil_image = img.convert('RGB')
            pil_image = pil_image.resize((self.ImageSizeX, self.ImageSizeY), Image.LANCZOS)
            pil_tensor = TF.to_tensor(pil_image)
            self.quantizedImage, *_ = self.vqganModel.encode(pil_tensor.to(self.vqganDevice).unsqueeze(0) * 2 - 1)
        else:
            one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=self.vqganDevice), n_toks).float()
            # z = one_hot @ vqganModel.quantize.embedding.weight
            if self.vqganGumbelEnabled:
                self.quantizedImage = one_hot @ self.vqganModel.quantize.embed.weight
            else:
                self.quantizedImage = one_hot @ self.vqganModel.quantize.embedding.weight

            self.quantizedImage = self.quantizedImage.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2) 
            #z = torch.rand_like(z)*2						# NR: check


        if self.init_weight:
            self.original_quantizedImage = self.quantizedImage.detach() #was clone() and setting grad to false before

        self.quantizedImage.requires_grad_(True)


    ########################
    ## cutout stuff during training
    ########################

    def GetSpatialPromptCutouts(self, cutouts, cutout_coords):
        cutouts_detached = cutouts.detach() #used to prevent gradient for unmask parts
        if self.blur_conv is not None:
            #Get the "blindfolded" image by blurring then addimg more noise
            facs = cutouts.new_empty([cutouts.size(0), 1, 1, 1]).uniform_(0, self.noise_fac)
            cutouts_blurred = self.blur_conv(cutouts_detached)+ facs * torch.randn_like(cutouts_detached)

        cutout_prompt_masks = []
        prompts_gradient_masked_cutouts = []

        for (x1,x2,y1,y2) in cutout_coords:

            promptMasks = []
            for prompt in self.embededPrompts:
                if prompt.promptMask == None:
                    continue

                promptMask = prompt.promptMask # color x h x w
                promptMasks.append(promptMask)

            promptMasks = torch.stack(promptMasks)

            keep_mask = promptMasks[:,:,y1:y2,x1:x2]
            keep_mask = ImageUtils.resample(keep_mask, (self.cut_size[0], self.cut_size[1])) 
                                    
            cutout_prompt_masks.append(keep_mask) # color x h x w


        cutout_prompt_masks = torch.stack(cutout_prompt_masks, dim=1) #-> prompts X cutouts X color X H X W

        idx:int = -1
        for prompt in self.embededPrompts: 

            if prompt.promptMask == None:
                prompts_gradient_masked_cutouts.append(cutouts)
                continue

            idx += 1

            keep_mask = cutout_prompt_masks[idx] #-> cutouts X color X H X W       
            #only apply this prompt if one image has a (big enough) part of mask
            if keep_mask.sum(dim=3).sum(dim=2).max()> self.cut_size[0]*2: #TODO: change this to a better test of overlap
            
                block_mask = 1-keep_mask

                #compose cutout of gradient and non-gradient parts
                if prompt.maskBlindfold and ((not isinstance(prompt.maskBlindfold,float)) or prompt.maskBlindfold>random.random()):
                    gradient_masked_cutouts = keep_mask*cutouts + block_mask*cutouts_blurred
                else:
                    gradient_masked_cutouts = keep_mask*cutouts + block_mask*cutouts_detached

                prompts_gradient_masked_cutouts.append(gradient_masked_cutouts)

        cutouts = torch.cat(prompts_gradient_masked_cutouts,dim=0)  

        return cutouts  


    def GetCutouts(self, synthedImage):
        cutouts, cutout_coords = self.CurrentCutoutMethod(synthedImage)

        # attempt masking stuff
        if self.use_spatial_prompts:
            if cutout_coords == None:
                print("Error! current cutout method does not support spatial prompts")
                raise NotImplementedError

            cutouts = self.GetSpatialPromptCutouts( cutouts, cutout_coords )
  

        return cutouts  # cutouts x prompts x clipW x clipH  # maybe its cutouts*prompts x channels x clipW x clipH


    def GetCutoutResults(self, clipEncodedImage, iteration:int):
        result = [] 

        if self.init_weight:
            # result.append(F.mse_loss(z, z_orig) * args.init_weight / 2)
            result.append(F.mse_loss(self.quantizedImage, torch.zeros_like(self.original_quantizedImage)) * ((1/torch.tensor(iteration*2 + 1))*self.init_weight) / 2)

        if self.use_spatial_prompts:
            for prompt_masked_iii,prompt in zip(torch.chunk(clipEncodedImage,len(self.embededPrompts),dim=0),self.embededPrompts):
                result.append(prompt(prompt_masked_iii))
        else:
            for prompt in self.embededPrompts:
                result.append(prompt(clipEncodedImage))      

        return result
