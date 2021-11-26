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
    def __init__(self, cmdArgs ):
        ### this will define all classwide member variables, so its easy to see
        ## should really convert this to something that is more explicit, but that will come later
        self.args = cmdArgs

        #### class wide variables set with default values
        self.clipPerceptorInputResolution = None # set after loading clip
        self.clipPerceptor = None # clip model
        self.clipDevice = None # torch device clip model is loaded onto
        self.cifar100 = None #one shot clip model classes, used when logging clip info

        self.quantizedImage = None # source image thats fed into taming transformers
        self.vqganDevice = None #torch device vqgan model is loaded onto
        self.vqganModel = None #vqgan model
        self.gumbel = False #vqgan gumbel model in use

        self.optimiser = None #currently in use optimiser        

        # cuts
        self.CurrentCutoutMethod = None

        # prompts
        self.embededPrompts = []
        self.all_phrases = []

        #### these need better names, wtf are they exactly?
        self.z_min = None
        self.z_max = None
        self.sideX = None
        self.sideY = None
        self.z_orig = None

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

        self.CurrentCutoutMethod = self.GetMakeCutouts( self.clipPerceptorInputResolution )
        
        # CLIP tokenize/encode   
        if self.args.prompts:
            for prompt in self.args.prompts:
                self.EmbedTextPrompt(prompt)

        for prompt in self.args.image_prompts:
            path, weight, stop = split_prompt(prompt)
            img = Image.open(path)
            pil_image = img.convert('RGB')
            img = imageUtils.resize_image(pil_image, (self.sideX, self.sideY))
            batch = self.CurrentCutoutMethod(TF.to_tensor(img).unsqueeze(0).to(self.clipDevice))
            embed = self.clipPerceptor.encode_image(self.normalize(batch)).float()
            self.embededPrompts.append(Prompt(embed, weight, stop).to(self.clipDevice))

        for seed, weight in zip(self.args.noise_prompt_seeds, self.args.noise_prompt_weights):
            gen = torch.Generator().manual_seed(seed)
            embed = torch.empty([1, self.clipPerceptor.visual.output_dim]).normal_(generator=gen)
            self.embededPrompts.append(Prompt(embed, weight).to(self.clipDevice))

        self.optimiser = self.get_optimiser(self.quantizedImage, self.args.optimiser,self.args.step_size)

        if self.args.optimiser == "MADGRAD":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimiser, 'min', factor=0.999, patience=0)  

        # Output for the user
        print('Optimising using:', self.args.optimiser)

        if self.args.prompts:
            print('Using text prompts:', self.args.prompts)  
        if self.args.image_prompts:
            print('Using image prompts:', self.args.image_prompts)
        if self.args.init_image:
            print('Using initial image:', self.args.init_image)
        if self.args.noise_prompt_weights:
            print('Noise prompt weights:', self.args.noise_prompt_weights)    


    # this method should reset everything aside from CLIP & vqgan model loading
    # this should allow this to run in 'server mode' where it can get repeated requests
    # to geenrate without having unload / reload models
    def PartialReset(self):
        # TODO
        pass


    # shutdown everything
    def Shutdown(self):
        # TODO
        pass


    ##############
    ##  Getters and converters
    ##############
    def GerCurrentImageAsPIL(self):
        out = self.synth(self.quantizedImage, self.gumbel)
        return TF.to_pil_image(out[0].cpu())

    def GetCurrentImageSynthed(self):
        return self.synth( self.quantizedImage, self.gumbel)

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

    def log_torch_mem(self):
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r-a  # free inside reserved
        print("total     VRAM:  " + str(t))
        print("reserved  VRAM:  " + str(r))
        print("allocated VRAM:  " + str(a))
        print("free      VRAM:  " + str(f))


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
        if self.args.cut_method == 'latest':
            self.args.cut_method = "nerdy"

        cutSize = self.args.cut_size
        if cutSize[0] == 0:
            cutSize[0] = clipPerceptorInputResolution

        if cutSize[1] == 0:
            cutSize[1] = clipPerceptorInputResolution    

        cutsMatchClip = True 
        if clipPerceptorInputResolution != cutSize or clipPerceptorInputResolution != cutSize[0] or clipPerceptorInputResolution != cutSize[1]:
            cutsMatchClip = False

        print("Cutouts method: " + self.args.cut_method + " using cutSize: " + str(cutSize) + '  Matches clipres: ' + str(cutsMatchClip))

        # used for whatever test cut thing im doing
        if self.args.cut_method == 'test':
            make_cutouts = makeCutouts.MakeCutoutsOneSpot(clipPerceptorInputResolution, cutSize[0], cutSize[1], self.args.cutn, cut_pow=self.args.cut_pow, use_pool=True, augments=self.args.augments)


        elif self.args.cut_method == 'growFromCenter':
            make_cutouts = makeCutouts.MakeCutoutsGrowFromCenter(clipPerceptorInputResolution, cutSize[0], cutSize[1], self.args.cutn, cut_pow=self.args.cut_pow, use_pool=True, augments=self.args.augments)        
        elif self.args.cut_method == 'squish':        
            make_cutouts = makeCutouts.MakeCutoutsSquish(clipPerceptorInputResolution, cutSize[0], cutSize[1], self.args.cutn, cut_pow=self.args.cut_pow, use_pool=True, augments=self.args.augments)
        elif self.args.cut_method == 'original':
            make_cutouts = makeCutouts.MakeCutoutsOrig(clipPerceptorInputResolution, self.args.cutn, cut_pow=self.args.cut_pow, augments=self.args.augments)
        elif self.args.cut_method == 'nerdy':
            make_cutouts = makeCutouts.MakeCutoutsNerdy(clipPerceptorInputResolution, self.args.cutn, cut_pow=self.args.cut_pow, augments=self.args.augments)
        elif self.args.cut_method == 'nerdyNoPool':
            make_cutouts = makeCutouts.MakeCutoutsNerdyNoPool(clipPerceptorInputResolution, self.args.cutn, cut_pow=self.args.cut_pow, augments=self.args.augments)
        else:
            print("Bad cut method selected")

        return make_cutouts



    ##########################
    ### One time init things... parsed from passed in args
    ##########################

    def InitTorch(self):
        print("Using pyTorch: " + str( torch.__version__) )
        print("Using mixed precision: " + str(self.args.use_mixed_precision) )  

        #TODO hacky as fuck
        makeCutouts.use_mixed_precision = self.args.use_mixed_precision

        if self.args.seed is None:
            seed = torch.seed()
        else:
            seed = self.args.seed  

        print('Using seed:', seed)
        self.seed_torch(seed)

        if self.args.deterministic >= 2:
            print("Determinism at max: forcing a lot of things so this will work, no augs, non-pooling cut method, bad resampling")
            self.args.augments = "none"
            self.args.cut_method = "original"

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
        elif self.args.deterministic == 1:
            print("Determinism at medium: cudnn determinism and benchmark disabled")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False # NR: True is a bit faster, but can lead to OOM. False is more deterministic.
        else:
            print("Determinism at minimum: cudnn benchmark on")
            torch.backends.cudnn.benchmark = True #apparently slightly faster, but less deterministic  

        if self.args.use_mixed_precision==True:
            print("cant use augments in mixed precision mode yet...")
            self.args.augments = [] 

        # Fallback to CPU if CUDA is not found and make sure GPU video rendering is also disabled
        # NB. May not work for AMD cards?
        if not self.args.cuda_device == 'cpu' and not torch.cuda.is_available():
            self.args.cuda_device = 'cpu'
            print("Warning: No GPU found! Using the CPU instead. The iterations will be slow.")
            print("Perhaps CUDA/ROCm or the right pytorch version is not properly installed?")     

        if self.args.anomaly_checker:
            torch.autograd.set_detect_anomaly(True)




    def InitPrompts(self):
        # Split text prompts using the pipe character (weights are split later)
        if self.args.prompts:
            # For stories, there will be many phrases
            story_phrases = [phrase.strip() for phrase in self.args.prompts.split("^")]
            
            # Make a list of all phrases
            for phrase in story_phrases:
                self.all_phrases.append(phrase.split("|"))
            
            # First phrase
            self.args.prompts = self.all_phrases[0]
            
        # Split target images using the pipe character (weights are split later)
        if self.args.image_prompts:
            self.args.image_prompts = self.args.image_prompts.split("|")
            self.args.image_prompts = [image.strip() for image in self.args.image_prompts]




    def InitClip(self):
        if self.args.log_clip:
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
        print("using clip model: " + self.args.clip_model)

        if self.args.clip_cpu == False:
            self.clipDevice = self.vqganDevice
            if jit == False:
                self.clipPerceptor = clip.load(self.args.clip_model, jit=jit, download_root="./clipModels/")[0].eval().requires_grad_(False).to(self.clipDevice)
            else:
                self.clipPerceptor = clip.load(self.args.clip_model, jit=jit, download_root="./clipModels/")[0].eval().to(self.clipDevice)    
        else:
            self.clipDevice = torch.device("cpu")
            self.clipPerceptor = clip.load(self.args.clip_model, "cpu", jit=jit)[0].eval().requires_grad_(False).to(self.clipDevice) 



        print("---  CLIP model loaded to " + str(self.clipDevice) +" ---")
        self.log_torch_mem()
        print("--- / CLIP model loaded ---")

        self.clipPerceptorInputResolution = self.clipPerceptor.visual.input_resolution



    def InitVQGAN(self):
        self.vqganDevice = torch.device(self.args.cuda_device)

        config_path = self.args.vqgan_config
        checkpoint_path = self.args.vqgan_checkpoint

        self.gumbel = False
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
            self.gumbel = True
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
        toksX, toksY = self.args.size[0] // vqganNumResolutionsF, self.args.size[1] // vqganNumResolutionsF
        sideX, sideY = toksX * vqganNumResolutionsF, toksY * vqganNumResolutionsF

        print("vqgan input resolutions: " + str(self.vqganModel.decoder.num_resolutions))
        print("cliperceptor input_res (aka cut size): " + str(self.clipPerceptorInputResolution) + " and whatever f is supposed to be: " + str(vqganNumResolutionsF))
        print("Toks X,Y: " + str(toksX) + ", " + str(toksY) + "      SizeX,Y: " + str(sideX) + ", " + str(sideY))
        
        # Gumbel or not?
        if self.gumbel:
            e_dim = 256
            n_toks = self.vqganModel.quantize.n_embed
            self.z_min = self.vqganModel.quantize.embed.weight.min(dim=0).values[None, :, None, None]
            self.z_max = self.vqganModel.quantize.embed.weight.max(dim=0).values[None, :, None, None]
        else:
            e_dim = self.vqganModel.quantize.e_dim
            n_toks = self.vqganModel.quantize.n_e
            self.z_min = self.vqganModel.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
            self.z_max = self.vqganModel.quantize.embedding.weight.max(dim=0).values[None, :, None, None]


        if self.args.init_image:
            if 'http' in self.args.init_image:
                img = Image.open(urlopen(self.args.init_image))
            else:
                img = Image.open(self.args.init_image)
                pil_image = img.convert('RGB')
                pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
                pil_tensor = TF.to_tensor(pil_image)
                self.quantizedImage, *_ = self.vqganModel.encode(pil_tensor.to(self.vqganDevice).unsqueeze(0) * 2 - 1)
        elif self.args.init_noise == 'pixels':
            img = imageUtils.random_noise_image(self.args.size[0], self.args.size[1])    
            pil_image = img.convert('RGB')
            pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
            pil_tensor = TF.to_tensor(pil_image)
            self.quantizedImage, *_ = self.vqganModel.encode(pil_tensor.to(self.vqganDevice).unsqueeze(0) * 2 - 1)
        elif self.args.init_noise == 'gradient':
            img = imageUtils.random_gradient_image(self.args.size[0], self.args.size[1])
            pil_image = img.convert('RGB')
            pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
            pil_tensor = TF.to_tensor(pil_image)
            self.quantizedImage, *_ = self.vqganModel.encode(pil_tensor.to(self.vqganDevice).unsqueeze(0) * 2 - 1)
        else:
            one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=self.vqganDevice), n_toks).float()
            # z = one_hot @ vqganModel.quantize.embedding.weight
            if self.gumbel:
                self.quantizedImage = one_hot @ self.vqganModel.quantize.embed.weight
            else:
                self.quantizedImage = one_hot @ self.vqganModel.quantize.embedding.weight

            self.quantizedImage = self.quantizedImage.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2) 
            #z = torch.rand_like(z)*2						# NR: check


        if self.args.init_weight:
            z_orig = self.quantizedImage.clone()
            z_orig.requires_grad_(False)

        self.quantizedImage.requires_grad_(True)


    ################################
    ## clip one shot analysis, just for fun, probably done wrong
    ###############################
    @torch.inference_mode()
    def WriteLogClipResults(self, imgout):
        #TODO properly manage initing the cifar100 stuff here if its not already

        img = self.normalize(self.CurrentCutoutMethod(imgout))

        if self.args.log_clip_oneshot:
            #one shot identification
            image_features = self.clipPerceptor.encode_image(img).float()

            text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in self.cifar100.classes]).to(self.clipDevice)
            
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
                print(f"{self.cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")

        if self.args.log_clip:
            # prompt matching percentages
            textins = []
            promptPartStrs = []
            if self.args.prompts:
                for prompt in self.args.prompts:
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
        with torch.cuda.amp.autocast(self.args.use_mixed_precision):

            cutouts = self.CurrentCutoutMethod(out)

            if self.clipDevice != self.vqganDevice:
                iii = self.clipPerceptor.encode_image(self.normalize(cutouts.to(self.clipDevice))).float()
            else:
                iii = self.clipPerceptor.encode_image(self.normalize(cutouts)).float()

            result = []        

            if self.args.init_weight:
                # result.append(F.mse_loss(z, z_orig) * args.init_weight / 2)
                result.append(F.mse_loss(self.quantizedImage, torch.zeros_like(self.z_orig)) * ((1/torch.tensor(iteration*2 + 1))*self.args.init_weight) / 2)

            for prompt in self.embededPrompts:
                result.append(prompt(iii))
            
            return result # return loss


    def train(self, iteration):
        with torch.cuda.amp.autocast(self.args.use_mixed_precision):
            self.optimiser.zero_grad(set_to_none=True)
            
            out = self.synth(self.quantizedImage, self.gumbel) 
            
            lossAll = self.ascend_txt(iteration, out)
            lossSum = sum(lossAll)

            if self.args.optimiser == "MADGRAD":
                self.loss_idx.append(lossSum.item())
                if iteration > 100: #use only 100 last looses to avg
                    avg_loss = sum(self.loss_idx[iteration-100:])/len(self.loss_idx[iteration-100:]) 
                else:
                    avg_loss = sum(self.loss_idx)/len(self.loss_idx)

                self.scheduler.step(avg_loss)
            
            if self.args.use_mixed_precision == False:
                lossSum.backward()
                self.optimiser.step()
            else:
                self.gradScaler.scale(lossSum).backward()
                self.gradScaler.step(self.optimiser)
                self.gradScaler.update()
            
            with torch.inference_mode():
                self.quantizedImage.copy_(self.quantizedImage.maximum(self.z_min).minimum(self.z_max))

            return out, lossAll, lossSum
    