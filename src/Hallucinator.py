import sys
import os
import random
import numpy as np


# shut off tqdm log spam by uncommenting the below
from tqdm import tqdm
import ImageMods
# from functools import partialmethod
# tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

import makeCutouts
import imageUtils
import GenerateJob

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







####################################################
# main class used to start up the vqgan clip stuff, and allow for interactable generation
# - basic usage can be seen from generate.py
#
# - advanced life cycle fatures involve resetting the network without reloading all the models
#    saving time on repeated generations
#
# - advanced manipulation by modifying masks, prompts, image 
#
# - command pattern to modify internals between training steps
#
####################################################

class Hallucinator:
    # TODO: dont use argparse args, use a config / json / something
    # using argparseargs for now due to being in the middle of a refactor
    def __init__(self, argparseArgs ):
        ### this will define all classwide member variables, so its easy to see
        ## should really convert this to something that is more explicit, but that will come later
        self.config = argparseArgs

        #### class wide variables set with default values
        self.clipPerceptorInputResolution = None # set after loading clip
        self.clipPerceptor = None # clip model
        self.clipDevice = None # torch device clip model is loaded onto
        self.clipCifar100 = None #one shot clip model classes, used when logging clip info

        self.vqganDevice = None #torch device vqgan model is loaded onto
        self.vqganModel = None #vqgan model
        self.vqganGumbelEnabled = False #vqgan gumbel model in use
        
        # From imagenet - Which is better?
        #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                        std=[0.26862954, 0.26130258, 0.27577711])



    ########################
    ## jobs and commands
    ########################
    def CreateNewGenerationJob(self, settings):
        newJob = GenerateJob.GenerationJob(self, settings)
        newJob.Initialize()
        return newJob


    #############
    ## Life cycle
    #############

    # does the minimal initialization that we shouldnt need to reset, unless we
    # force a change in clip/torch/vqgan models
    def Initialize(self):
        self.InitTorch()        
        self.InitVQGAN()
        self.InitClip()
        print('Using vqgandevice:', self.vqganDevice)
        print('Using clipdevice:', self.clipDevice)



    ##############
    ##  Getters and converters
    ##############
    def GerCurrentImageAsPIL(self, genJob):
        out = self.synth(genJob.quantizedImage, genJob.vqganGumbelEnabled)
        return TF.to_pil_image(out[0].cpu())

    def GetCurrentImageSynthed(self, genJob):
        return self.synth( genJob.quantizedImage, genJob.vqganGumbelEnabled)

    def ConvertToPIL(self, synthedImage):
        return TF.to_pil_image(synthedImage[0].cpu())


    ##################
    ### Logging and other internal helper methods...
    ##################
    def seed_torch(self, seed=42):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

    def log_torch_mem(self, title = ''):
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r-a  # free inside reserved

        if title != '':
            print('>>>>  ' + title)

        print("total     VRAM:  " + str(t))
        print("reserved  VRAM:  " + str(r))
        print("allocated VRAM:  " + str(a))
        print("free      VRAM:  " + str(f))

        if title != '':
            print('>>>>  /' + title)


    ###################
    # Vector quantize
    ###################
    def vector_quantize(self, x, codebook):
        d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
        indices = d.argmin(-1)
        x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
        return replace_grad(x_q, x)

    def synth(self, z, gumbelMode):
        if gumbelMode:
            z_q = self.vector_quantize(z.movedim(1, 3), self.vqganModel.quantize.embed.weight).movedim(3, 1)
        else:
            z_q = self.vector_quantize(z.movedim(1, 3), self.vqganModel.quantize.embedding.weight).movedim(3, 1)
        return makeCutouts.clamp_with_grad(self.vqganModel.decode(z_q).add(1).div(2), 0, 1)



    ########################
    # get the optimiser ###
    ########################
    def get_optimiser(self, quantizedImg, opt_name, opt_lr):

        # from nerdy project, potential learning rate tweaks?
        # Messing with learning rate / optimisers
        #variable_lr = args.step_size
        #optimiser_list = [['Adam',0.075],['AdamW',0.125],['Adagrad',0.2],['Adamax',0.125],['DiffGrad',0.075],['RAdam',0.125],['RMSprop',0.02]]


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
            print("Unknown optimiser. Are choices broken?")
            opt = optim.Adam([quantizedImg], lr=opt_lr)
        return opt



    ###################
    ##  Get an instance of the cutout we are goign to use
    ###################
    def GetMakeCutouts( self, clipPerceptorInputResolution ):
        # Cutout class options:
        # 'squish', 'latest','original','updated' or 'updatedpooling'
        if self.config.cut_method == 'latest':
            self.config.cut_method = "nerdy"

        cutSize = self.config.cut_size
        if cutSize[0] == 0:
            cutSize[0] = clipPerceptorInputResolution

        if cutSize[1] == 0:
            cutSize[1] = clipPerceptorInputResolution    

        cutsMatchClip = True 
        if clipPerceptorInputResolution != cutSize or clipPerceptorInputResolution != cutSize[0] or clipPerceptorInputResolution != cutSize[1]:
            cutsMatchClip = False

        print("Cutouts method: " + self.config.cut_method + " using cutSize: " + str(cutSize) + '  Matches clipres: ' + str(cutsMatchClip))

        # used for whatever test cut thing im doing
        if self.config.cut_method == 'test':
            make_cutouts = makeCutouts.MakeCutoutsOneSpot(clipPerceptorInputResolution, cutSize[0], cutSize[1], self.config.cutn, cut_pow=self.config.cut_pow, use_pool=True, augments=self.config.augments)


        elif self.config.cut_method == 'maskTest':
            make_cutouts = makeCutouts.MakeCutoutsMaskTest(clipPerceptorInputResolution, cutSize[0], cutSize[1], self.config.cutn, cut_pow=self.config.cut_pow, use_pool=False, augments=[])



        elif self.config.cut_method == 'growFromCenter':
            make_cutouts = makeCutouts.MakeCutoutsGrowFromCenter(clipPerceptorInputResolution, cutSize[0], cutSize[1], self.config.cutn, cut_pow=self.config.cut_pow, use_pool=True, augments=self.config.augments)        
        elif self.config.cut_method == 'squish':        
            make_cutouts = makeCutouts.MakeCutoutsSquish(clipPerceptorInputResolution, cutSize[0], cutSize[1], self.config.cutn, cut_pow=self.config.cut_pow, use_pool=True, augments=self.config.augments)
        elif self.config.cut_method == 'original':
            make_cutouts = makeCutouts.MakeCutoutsOrig(clipPerceptorInputResolution, self.config.cutn, cut_pow=self.config.cut_pow, augments=self.config.augments)
        elif self.config.cut_method == 'nerdy':
            make_cutouts = makeCutouts.MakeCutoutsNerdy(clipPerceptorInputResolution, self.config.cutn, cut_pow=self.config.cut_pow, augments=self.config.augments)
        elif self.config.cut_method == 'nerdyNoPool':
            make_cutouts = makeCutouts.MakeCutoutsNerdyNoPool(clipPerceptorInputResolution, self.config.cutn, cut_pow=self.config.cut_pow, augments=self.config.augments)
        else:
            print("Bad cut method selected")

        return make_cutouts



    ##########################
    ### One time init things... parsed from passed in args
    ##########################

    def InitTorch(self):
        print("Using pyTorch: " + str( torch.__version__) )
        print("Using mixed precision: " + str(self.config.use_mixed_precision) )  

        #TODO hacky as fuck
        makeCutouts.use_mixed_precision = self.config.use_mixed_precision

        if self.config.seed is None:
            seed = torch.seed()
        else:
            seed = self.config.seed  

        print('Using seed:', seed)
        self.seed_torch(seed)

        if self.config.deterministic >= 2:
            print("Determinism at max: forcing a lot of things so this will work, no augs, non-pooling cut method, bad resampling")
            self.config.augments = "none"
            self.config.cut_method = "original"

            # need to make cutouts use deterministic stuff... probably not a good way
            makeCutouts.deterministic = True

            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False # NR: True is a bit faster, but can lead to OOM. False is more deterministic.

            torch.use_deterministic_algorithms(True)	   # NR: grid_sampler_2d_backward_cuda does not have a deterministic implementation   

            # CUBLAS determinism:
            # Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, 
            # but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, 
            # you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. 
            # For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility  
            #   
            # set a debug environment variable CUBLAS_WORKSPACE_CONFIG to ":16:8" (may limit overall performance) or ":4096:8" (will increase library footprint in GPU memory by approximately 24MiB).
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            #EXPORT CUBLAS_WORKSPACE_CONFIG=:4096:8

            # from nightly build for 1.11 -> 0 no warn, 1 warn, 2 error
            # torch.set_deterministic_debug_mode(2)
        elif self.config.deterministic == 1:
            print("Determinism at medium: cudnn determinism and benchmark disabled")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False # NR: True is a bit faster, but can lead to OOM. False is more deterministic.
        else:
            print("Determinism at minimum: cudnn benchmark on")
            torch.backends.cudnn.benchmark = True #apparently slightly faster, but less deterministic  

        if self.config.use_mixed_precision==True:
            print("cant use augments in mixed precision mode yet...")
            self.config.augments = [] 

        # Fallback to CPU if CUDA is not found and make sure GPU video rendering is also disabled
        # NB. May not work for AMD cards?
        if not self.config.cuda_device == 'cpu' and not torch.cuda.is_available():
            self.config.cuda_device = 'cpu'
            print("Warning: No GPU found! Using the CPU instead. The iterations will be slow.")
            print("Perhaps CUDA/ROCm or the right pytorch version is not properly installed?")     

        if self.config.anomaly_checker:
            torch.autograd.set_detect_anomaly(True)


    def InitClip(self):
        if self.config.log_clip:
            print("logging clip probabilities at end, loading vocab stuff")
            cifar100 = CIFAR100(root=".", download=True, train=False)

        jit = False
        try:
            # try here, since using nightly build of pytorch has a version scheme like dev23723h
            if [int(n) for n in torch.__version__.split(".")] < [1, 8, 1]:
                jit = True
        except:
            jit = False

        print( "available clip models: " + str(clip.available_models() ))
        print("CLIP jit: " + str(jit))
        print("using clip model: " + self.config.clip_model)

        if self.config.clip_cpu == False:
            self.clipDevice = self.vqganDevice
            if jit == False:
                self.clipPerceptor = clip.load(self.config.clip_model, jit=jit, download_root="./clipModels/")[0].eval().requires_grad_(False).to(self.clipDevice)
            else:
                self.clipPerceptor = clip.load(self.config.clip_model, jit=jit, download_root="./clipModels/")[0].eval().to(self.clipDevice)    
        else:
            self.clipDevice = torch.device("cpu")
            self.clipPerceptor = clip.load(self.config.clip_model, "cpu", jit=jit)[0].eval().requires_grad_(False).to(self.clipDevice) 



        print("---  CLIP model loaded to " + str(self.clipDevice) +" ---")
        self.log_torch_mem()
        print("--- / CLIP model loaded ---")

        self.clipPerceptorInputResolution = self.clipPerceptor.visual.input_resolution



    def InitVQGAN(self):
        self.vqganDevice = torch.device(self.config.cuda_device)

        config_path = self.config.vqgan_config
        checkpoint_path = self.config.vqgan_checkpoint

        self.vqganGumbelEnabled = False
        config = OmegaConf.load(config_path)

        print("---  VQGAN config " + str(config_path))    
        print(yaml.dump(OmegaConf.to_container(config)))
        print("---  / VQGAN config " + str(config_path))

        if config.model.target == 'taming.models.vqgan.VQModel':
            self.vqganModel = vqgan.VQModel(**config.model.params)
            self.vqganModel.eval().requires_grad_(False)
            self.vqganModel.init_from_ckpt(checkpoint_path)
        elif config.model.target == 'taming.models.vqgan.GumbelVQ':
            self.vqganModel = vqgan.GumbelVQ(**config.model.params)
            self.vqganModel.eval().requires_grad_(False)
            self.vqganModel.init_from_ckpt(checkpoint_path)
            self.vqganGumbelEnabled = True
        elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
            parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
            parent_model.eval().requires_grad_(False)
            parent_model.init_from_ckpt(checkpoint_path)
            self.vqganModel = parent_model.first_stage_model
        else:
            raise ValueError(f'unknown model type: {config.model.target}')
        del self.vqganModel.loss   

        self.vqganModel.to(self.vqganDevice)

        print("---  VQGAN model loaded ---")
        self.log_torch_mem()
        print("--- / VQGAN model loaded ---")


    ################################
    ## clip one shot analysis, just for fun, probably done wrong
    ###############################
    @torch.inference_mode()
    def WriteLogClipResults(self, imgout):
        #TODO properly manage initing the cifar100 stuff here if its not already

        img = self.normalize(self.CurrentCutoutMethod(imgout))

        if self.config.log_clip_oneshot:
            #one shot identification
            image_features = self.clipPerceptor.encode_image(img).float()

            text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in self.clipCifar100.classes]).to(self.clipDevice)
            
            text_features = self.clipPerceptor.encode_text(text_inputs).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Pick the top 5 most similar labels for the image
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values, indices = similarity[0].topk(5)

            # Print the result
            print("\nOne-shot predictions:\n")
            for value, index in zip(values, indices):
                print(f"{self.clipCifar100.classes[index]:>16s}: {100 * value.item():.2f}%")

        if self.config.log_clip:
            # prompt matching percentages
            textins = []
            promptPartStrs = []
            if self.config.prompts:
                for prompt in self.config.prompts:
                    txt, weight, stop = split_prompt(prompt)  
                    splitTxt = txt.split()
                    for stxt in splitTxt:   
                        promptPartStrs.append(stxt)       
                        textins.append(clip.tokenize(stxt))
                    for i in range(len(splitTxt) - 1):
                        promptPartStrs.append(splitTxt[i] + " " + splitTxt[i + 1])       
                        textins.append(clip.tokenize(splitTxt[i] + " " + splitTxt[i + 1]))
                    for i in range(len(splitTxt) - 2):
                        promptPartStrs.append(splitTxt[i] + " " + splitTxt[i + 1] + " " + splitTxt[i + 2])       
                        textins.append(clip.tokenize(splitTxt[i] + " " + splitTxt[i + 1] + " " + splitTxt[i + 2]))                    

            text_inputs = torch.cat(textins).to(self.clipDevice)
            
            image_features = self.clipPerceptor.encode_image(img).float()
            text_features = self.clipPerceptor.encode_text(text_inputs).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Pick the top 5 most similar labels for the image
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            top = 5
            if top > similarity[0].size()[0]:
                top = similarity[0].size()[0]

            values, indices = similarity[0].topk(top)

            # Print the result
            print("\nPrompt matching predictions:\n")
            for value, index in zip(values, indices):        
                print(f"{promptPartStrs[index]:>16s}: {100 * value.item():.2f}%")   



    ######################
    ### interactive generation steps and training
    ######################

    def ascend_txt(self, genJob, iteration, synthedImage):
        with torch.cuda.amp.autocast(self.config.use_mixed_precision):

            cutouts, cutout_coords = genJob.CurrentCutoutMethod(synthedImage)

            # attempt masking stuff
            if genJob.config.use_spatial_prompts:
                cutouts_detached = cutouts.detach() #used to prevent gradient for unmask parts
                if genJob.blur_conv is not None:
                    #Get the "blindfolded" image by blurring then addimg more noise
                    facs = cutouts.new_empty([cutouts.size(0), 1, 1, 1]).uniform_(0, genJob.noise_fac)
                    cutouts_blurred = genJob.blur_conv(cutouts_detached)+ facs * torch.randn_like(cutouts_detached)


                cut_size = genJob.config.cut_size

                #get mask patches
                cutout_prompt_masks = []
                for (x1,x2,y1,y2) in cutout_coords:
                    cutout_mask = genJob.prompt_masks[:,:,y1:y2,x1:x2]
                    cutout_mask = makeCutouts.resample(cutout_mask, (cut_size[0], cut_size[1]))
                    cutout_prompt_masks.append(cutout_mask)
                cutout_prompt_masks = torch.stack(cutout_prompt_masks,dim=1) #-> prompts X cutouts X color X H X W
                
                #apply each prompt, masking gradients
                prompts_gradient_masked_cutouts = []
                for idx,prompt in enumerate(genJob.embededPrompts):
                    keep_mask = cutout_prompt_masks[idx] #-> cutouts X color X H X W
                    #only apply this prompt if one image has a (big enough) part of mask
                    if keep_mask.sum(dim=3).sum(dim=2).max()> cut_size[0]*2: #todo, change this
                        
                        block_mask = 1-keep_mask

                        #compose cutout of gradient and non-gradient parts
                        if genJob.blindfold[idx] and ((not isinstance(genJob.blindfold[idx],float)) or genJob.blindfold[idx]>random.random()):
                            gradient_masked_cutouts = keep_mask*cutouts + block_mask*cutouts_blurred
                        else:
                            gradient_masked_cutouts = keep_mask*cutouts + block_mask*cutouts_detached

                        prompts_gradient_masked_cutouts.append(gradient_masked_cutouts)
                cutouts = torch.cat(prompts_gradient_masked_cutouts,dim=0)            



            if self.clipDevice != self.vqganDevice:
                clipEncodedImage = self.clipPerceptor.encode_image(self.normalize(cutouts.to(self.clipDevice))).float()
            else:
                clipEncodedImage = self.clipPerceptor.encode_image(self.normalize(cutouts)).float()

            result = []        

            if genJob.config.init_weight:
                # result.append(F.mse_loss(z, z_orig) * args.init_weight / 2)
                result.append(F.mse_loss(genJob.quantizedImage, torch.zeros_like(genJob.original_quantizedImage)) * ((1/torch.tensor(iteration*2 + 1))*genJob.config.init_weight) / 2)

            if genJob.config.use_spatial_prompts:
                for prompt_masked_iii,prompt in zip(torch.chunk(clipEncodedImage,genJob.num_prompts,dim=0),genJob.embededPrompts):
                    result.append(prompt(prompt_masked_iii))
            else:
                for prompt in genJob.embededPrompts:
                    result.append(prompt(clipEncodedImage))
            
            return result # return loss


    def train(self, genJob, iteration):
        with torch.cuda.amp.autocast(self.config.use_mixed_precision):
            genJob.optimiser.zero_grad(set_to_none=True)
            
            synthedImage = self.synth(genJob.quantizedImage, genJob.vqganGumbelEnabled) 
            
            lossAll = self.ascend_txt(genJob, iteration, synthedImage)
            lossSum = sum(lossAll)

            if genJob.config.optimiser == "MADGRAD":
                genJob.loss_idx.append(lossSum.item())
                if iteration > 100: #use only 100 last looses to avg
                    avg_loss = sum(self.loss_idx[iteration-100:])/len(self.loss_idx[iteration-100:]) 
                else:
                    avg_loss = sum(self.loss_idx)/len(self.loss_idx)

                genJob.scheduler.step(avg_loss)
            
            if self.config.use_mixed_precision == False:
                lossSum.backward()
                genJob.optimiser.step()
            else:
                genJob.gradScaler.scale(lossSum).backward()
                genJob.gradScaler.step(genJob.optimiser)
                genJob.gradScaler.update()
            
            with torch.inference_mode():
                genJob.quantizedImage.copy_(genJob.quantizedImage.maximum(genJob.z_min).minimum(genJob.z_max))

            return synthedImage, lossAll, lossSum



    #########################
    ### do manipulations to the image sent to vqgan prior to training steps
    ### for example, image mask lock, or the image zooming effect
    #########################
    def OnPreTrain(self, genJob, iteration):
        for mod in genJob.ImageModifiers:
            if mod.ShouldApply( ImageMods.ImageModStage.PreTrain, iteration ):
                mod.OnPreTrain( iteration )

    def OnFinishGeneration(self, genJob, iteration):
        for mod in genJob.ImageModifiers:
            if mod.ShouldApply( ImageMods.ImageModStage.FinishedGeneration, iteration ):
                mod.OnPreTrain( iteration )

    