import sys
import os
import random
import numpy as np


# shut off tqdm log spam by uncommenting the below
from tqdm import tqdm
# from functools import partialmethod
# tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

import makeCutouts
import imageUtils


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

        self.quantizedImage = None # source image thats fed into taming transformers
        self.vqganDevice = None #torch device vqgan model is loaded onto
        self.vqganModel = None #vqgan model
        self.vqganGumbelEnabled = False #vqgan gumbel model in use

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
        self.sideX = None
        self.sideY = None
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


        #lock image mask     
        self.lock_image_mask_np = None
        self.lock_image_original_pil = None

        self.lock_image_original_tensor = None
        self.lock_image_mask_tensor = None
        self.lock_image_mask_tensor_invert = None


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


    # does the old version of init, which does everything for a one shot generation
    def FullInitialize(self):
        self.MinimalInitialize()
        self.PerRequestInitialization()


    # does the minimal initialization that we shouldnt need to reset, unless we
    # force a change in clip/torch/vqgan models
    def MinimalInitialize(self):
        self.InitTorch()        
        self.InitVQGAN()
        self.InitClip()
        print('Using vqgandevice:', self.vqganDevice)
        print('Using clipdevice:', self.clipDevice)



    # stuff that needs to be initialized per request / reset between requests
    def PerRequestInitialization(self):        
        self.InitPrompts()
        self.InitStartingImage()
        self.InitSpatialPromptMasks()
        self.InitLockImageMask()

        self.CurrentCutoutMethod = self.GetMakeCutouts( self.clipPerceptorInputResolution )
        
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
            img = imageUtils.resize_image(pil_image, (self.sideX, self.sideY))
            batch = self.CurrentCutoutMethod(TF.to_tensor(img).unsqueeze(0).to(self.clipDevice))
            embed = self.clipPerceptor.encode_image(self.normalize(batch)).float()
            self.embededPrompts.append(Prompt(embed, weight, stop).to(self.clipDevice))

        for seed, weight in zip(self.config.noise_prompt_seeds, self.config.noise_prompt_weights):
            gen = torch.Generator().manual_seed(seed)
            embed = torch.empty([1, self.clipPerceptor.visual.output_dim]).normal_(generator=gen)
            self.embededPrompts.append(Prompt(embed, weight).to(self.clipDevice))

        self.optimiser = self.get_optimiser(self.quantizedImage, self.config.optimiser,self.config.step_size)

        if self.config.optimiser == "MADGRAD":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimiser, 'min', factor=0.999, patience=0)  

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
    def PartialReset(self):
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
        self.sideX = None
        self.sideY = None
        self.original_quantizedImage = None
        self.loss_idx = []
        self.scheduler = None

        gc.collect()
        torch.cuda.empty_cache()

        self.log_torch_mem('Post PartialReset')



    # shutdown everything
    def Shutdown(self):
        # TODO
        pass


    ##############
    ##  Getters and converters
    ##############
    def GerCurrentImageAsPIL(self):
        out = self.synth(self.quantizedImage, self.vqganGumbelEnabled)
        return TF.to_pil_image(out[0].cpu())

    def GetCurrentImageSynthed(self):
        return self.synth( self.quantizedImage, self.vqganGumbelEnabled)

    def ConvertToPIL(self, synthedImage):
        return TF.to_pil_image(synthedImage[0].cpu())


    #####################
    ### Helper type methods
    #####################

    def EmbedTextPrompt(self, prompt):
        txt, weight, stop = split_prompt(prompt)
        embed = self.clipPerceptor.encode_text(clip.tokenize(txt).to(self.clipDevice)).float()
        self.embededPrompts.append(Prompt(embed, weight, stop).to(self.clipDevice))



    #################
    ## stop and resume with trained weights...
    #################

    def ResetTraining(self):
        #TODO
        pass

    def SaveTrainedWeights(self):
        #TODO
        pass

    def LoadTrainedWeights(self):
        #TODO
        pass


    #################
    ## ways to interactively play with generation
    #################

    # sends in a command object that updates/modifies the state of the generation
    # between training steps
    def ProcessCommand( hallucinatorCmd):
        #TODO
        pass


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



    def InitStartingImage(self):
        vqganNumResolutionsF = 2**(self.vqganModel.decoder.num_resolutions - 1)
        self.toksX, self.toksY = self.config.size[0] // vqganNumResolutionsF, self.config.size[1] // vqganNumResolutionsF
        self.sideX, self.sideY = self.toksX * vqganNumResolutionsF, self.toksY * vqganNumResolutionsF

        print("vqgan input resolutions: " + str(self.vqganModel.decoder.num_resolutions))
        print("cliperceptor input_res (aka cut size): " + str(self.clipPerceptorInputResolution) + " and whatever f is supposed to be: " + str(vqganNumResolutionsF))
        print("Toks X,Y: " + str(self.toksX) + ", " + str(self.toksY) + "      SizeX,Y: " + str(self.sideX) + ", " + str(self.sideY))
        
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
                pil_image = pil_image.resize((self.sideX, self.sideY), Image.LANCZOS)
                pil_tensor = TF.to_tensor(pil_image)
                print( 'first encoding -> pil_tensor size: ' + str( pil_tensor.size() ) )
                self.quantizedImage, *_ = self.vqganModel.encode(pil_tensor.to(self.vqganDevice).unsqueeze(0) * 2 - 1)
        elif self.config.init_noise == 'pixels':
            img = imageUtils.random_noise_image(self.config.size[0], self.config.size[1])    
            pil_image = img.convert('RGB')
            pil_image = pil_image.resize((self.sideX, self.sideY), Image.LANCZOS)
            pil_tensor = TF.to_tensor(pil_image)
            self.quantizedImage, *_ = self.vqganModel.encode(pil_tensor.to(self.vqganDevice).unsqueeze(0) * 2 - 1)
        elif self.config.init_noise == 'gradient':
            img = imageUtils.random_gradient_image(self.config.size[0], self.config.size[1])
            pil_image = img.convert('RGB')
            pil_image = pil_image.resize((self.sideX, self.sideY), Image.LANCZOS)
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


        if self.config.init_weight or self.config.use_image_lock_mask:
            #TODO is this right?
            self.original_quantizedImage = self.quantizedImage.detach()
            #self.original_quantizedImage = self.quantizedImage.clone()
            #self.original_quantizedImage.requires_grad_(False)

        self.quantizedImage.requires_grad_(True)


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

    def ascend_txt(self, iteration, out):
        with torch.cuda.amp.autocast(self.config.use_mixed_precision):

            cutouts, cutout_coords = self.CurrentCutoutMethod(out)

            # attempt masking stuff
            if self.config.use_spatial_prompts:
                cutouts_detached = cutouts.detach() #used to prevent gradient for unmask parts
                if self.blur_conv is not None:
                    #Get the "blindfolded" image by blurring then addimg more noise
                    facs = cutouts.new_empty([cutouts.size(0), 1, 1, 1]).uniform_(0, self.noise_fac)
                    cutouts_blurred = self.blur_conv(cutouts_detached)+ facs * torch.randn_like(cutouts_detached)


                cut_size = self.config.cut_size

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



            if self.clipDevice != self.vqganDevice:
                clipEncodedImage = self.clipPerceptor.encode_image(self.normalize(cutouts.to(self.clipDevice))).float()
            else:
                clipEncodedImage = self.clipPerceptor.encode_image(self.normalize(cutouts)).float()

            result = []        

            if self.config.init_weight:
                # result.append(F.mse_loss(z, z_orig) * args.init_weight / 2)
                result.append(F.mse_loss(self.quantizedImage, torch.zeros_like(self.original_quantizedImage)) * ((1/torch.tensor(iteration*2 + 1))*self.config.init_weight) / 2)

            if self.config.use_spatial_prompts:
                for prompt_masked_iii,prompt in zip(torch.chunk(clipEncodedImage,self.num_prompts,dim=0),self.embededPrompts):
                    result.append(prompt(prompt_masked_iii))
            else:
                for prompt in self.embededPrompts:
                    result.append(prompt(clipEncodedImage))
            
            return result # return loss


    def train(self, iteration):
        with torch.cuda.amp.autocast(self.config.use_mixed_precision):
            self.optimiser.zero_grad(set_to_none=True)
            
            out = self.synth(self.quantizedImage, self.vqganGumbelEnabled) 
            
            lossAll = self.ascend_txt(iteration, out)
            lossSum = sum(lossAll)

            if self.config.optimiser == "MADGRAD":
                self.loss_idx.append(lossSum.item())
                if iteration > 100: #use only 100 last looses to avg
                    avg_loss = sum(self.loss_idx[iteration-100:])/len(self.loss_idx[iteration-100:]) 
                else:
                    avg_loss = sum(self.loss_idx)/len(self.loss_idx)

                self.scheduler.step(avg_loss)
            
            if self.config.use_mixed_precision == False:
                lossSum.backward()
                self.optimiser.step()
            else:
                self.gradScaler.scale(lossSum).backward()
                self.gradScaler.step(self.optimiser)
                self.gradScaler.update()
            
            with torch.inference_mode():
                self.quantizedImage.copy_(self.quantizedImage.maximum(self.z_min).minimum(self.z_max))

            return out, lossAll, lossSum



    #########################
    ### do manipulations to the image sent to vqgan prior to training steps
    ### for example, image mask lock, or the image zooming effect
    #########################
    def OnPreTrain(self, iteration):
        #TODO: make this use predicates or classes for manipulations, for now jsut hard code some
        # i am also 100% sure theres a better way to do this, but lets just see what happens for now
        #this is also very wrong, just see what happens...
        if self.config.use_image_lock_mask == True and iteration % self.config.image_lock_overwrite_iteration == 0:
            with torch.inference_mode():
                curQuantImg = self.synth(self.quantizedImage, self.vqganGumbelEnabled)

                #this removes the first dim sized 1 to match the rest
                curQuantImg = torch.squeeze(curQuantImg)

                #keepCurrentImg = torch.zeros(curQuantImg.size()) + 0
                keepCurrentImg = curQuantImg * self.lock_image_mask_tensor_invert.int().float()

                #keepOrig = torch.zeros(self.lock_image_original_tensor.size()) + 0
                keepOrig = self.lock_image_original_tensor * self.lock_image_mask_tensor.int().float()

                pil_tensor = keepCurrentImg + keepOrig

            # Re-encode
            self.quantizedImage, *_ = self.vqganModel.encode(pil_tensor.to(self.vqganDevice).unsqueeze(0) * 2 - 1)
            self.original_quantizedImage = self.quantizedImage.detach()
            
            self.quantizedImage.requires_grad_(True)
            self.optimiser = self.get_optimiser(self.quantizedImage, self.config.optimiser, self.config.step_size)

    def OnFinishGeneration(self):
        pass

            



    def InitLockImageMask(self):
        if self.config.use_image_lock_mask == False:
            return

        #store original
        self.lock_image_original_pil = self.GerCurrentImageAsPIL()
        self.lock_image_original_tensor = TF.to_tensor(self.lock_image_original_pil).to(self.vqganDevice)

        #Make prompt masks
        img = Image.open(self.config.image_lock_mask)
        pil_image = img.convert('RGB')
        #dest_gray = pil_image.convert('L')
        
        self.lock_image_mask_np  = np.asarray(pil_image)

        #makes float32 mask
        self.lock_image_mask_tensor = TF.to_tensor(self.lock_image_mask_np).to(self.vqganDevice)

        #make boolean masks
        self.lock_image_mask_tensor_invert = torch.logical_not( self.lock_image_mask_tensor )
        self.lock_image_mask_tensor = torch.logical_not( self.lock_image_mask_tensor_invert )
    