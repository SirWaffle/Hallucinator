import sys
import os
import random
from typing import Any, Tuple
import numpy as np
from tqdm import tqdm

from src import GenerationCommands
from src import GenerationCommand
from src import MakeCutouts
from src import GenerateJob

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
from taming.modules.diffusionmodules import model



import yaml
from urllib.request import urlopen

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

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


from torchvision.datasets import CIFAR100



####################################################
# main class used to start up the vqgan clip stuff, and allow for interactable generation
# - basic usage can be seen from generate.py
####################################################

class Hallucinator:

    def __init__(self, clipModel:str = 'ViT-B/32', vqgan_config_path:str = 'checkpoints/vqgan_imagenet_f16_16384.yaml', vqgan_checkpoint_path:str = 'checkpoints/vqgan_imagenet_f16_16384.ckpt', 
                 use_mixed_precision:bool = False, clip_cpu:bool = False, randomSeed:int = None, cuda_device:str = "cuda:0", anomaly_checker:bool = False, deterministic:int = 1, 
                 log_clip:bool = False, log_clip_oneshot:bool = False, log_mem:bool = False, display_freq:int = 50 ):

        ## passed in settings
        self.clip_model = clipModel
        self.vqgan_config_path = vqgan_config_path
        self.vqgan_checkpoint_path = vqgan_checkpoint_path
        self.use_mixed_precision = use_mixed_precision
        self.clip_cpu = clip_cpu
        self.seed = randomSeed
        self.cuda_device = cuda_device
        self.anomaly_checker = anomaly_checker
        self.deterministic = deterministic
        self.log_clip = log_clip
        self.log_clip_oneshot = log_clip_oneshot
        self.log_mem = log_mem
        self.display_freq = display_freq

        #### class wide variables set with default values
        self.clipPerceptorInputResolution = None # set after loading clip
        self.clipPerceptor: model.CLIP = None # clip model
        self.clipDevice = None # torch device clip model is loaded onto
        self.clipCifar100 = None #one shot clip model classes, used when logging clip info

        self.vqganDevice = None #torch device vqgan model is loaded onto
        self.vqganModel: vqgan.VQModel = None #vqgan model
        self.vqganGumbelEnabled = False #vqgan gumbel model in use
        
        # From imagenet - Which is better?
        #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                        std=[0.26862954, 0.26130258, 0.27577711])


        # terrible hack
        model.do_nan_check = self.use_mixed_precision

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

    ##################
    ### Logging and other internal helper methods...
    ##################
    def seed_torch(self, seed:int=42):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

    def log_torch_mem(self, title:str = ''):
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
    def vector_quantize(self, x, codebook) -> torch.Tensor:
        d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
        indices = d.argmin(-1)
        x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
        return GenerateJob.replace_grad(x_q, x)

    def synth(self, z, gumbelMode) -> torch.Tensor:
        if gumbelMode:
            z_q = self.vector_quantize(z.movedim(1, 3), self.vqganModel.quantize.embed.weight).movedim(3, 1)
        else:
            z_q = self.vector_quantize(z.movedim(1, 3), self.vqganModel.quantize.embedding.weight).movedim(3, 1)
        return MakeCutouts.clamp_with_grad(self.vqganModel.decode(z_q).add(1).div(2), 0, 1)



    ##########################
    ### One time init things... parsed from passed in args
    ##########################

    def InitTorch(self):
        print("Using pyTorch: " + str( torch.__version__) )
        print("Using mixed precision: " + str(self.use_mixed_precision) )  

        #TODO hacky as fuck
        MakeCutouts.use_mixed_precision = self.use_mixed_precision

        if self.seed is None:
            self.seed = torch.seed()

        print('Using seed:', self.seed)
        self.seed_torch(self.seed)

        if self.deterministic >= 2:
            print("Determinism at max: forcing a lot of things so this will work, no augs, non-pooling cut method, bad resampling")

            # need to make cutouts use deterministic stuff... probably not a good way
            MakeCutouts.deterministic = True

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
        elif self.deterministic == 1:
            print("Determinism at medium: cudnn determinism and benchmark disabled")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False # NR: True is a bit faster, but can lead to OOM. False is more deterministic.
        else:
            print("Determinism at minimum: cudnn benchmark on")
            torch.backends.cudnn.benchmark = True #apparently slightly faster, but less deterministic  

        if self.use_mixed_precision==True:
            print("Hallucinator mxed precision mode enabled: cant use augments in mixed precision mode yet")

        # Fallback to CPU if CUDA is not found and make sure GPU video rendering is also disabled
        # NB. May not work for AMD cards?
        if not self.cuda_device == 'cpu' and not torch.cuda.is_available():
            self.cuda_device = 'cpu'
            print("Warning: No GPU found! Using the CPU instead. The iterations will be slow.")
            print("Perhaps CUDA/ROCm or the right pytorch version is not properly installed?")     

        if self.anomaly_checker:
            torch.autograd.set_detect_anomaly(True)


    def InitClip(self):
        if self.log_clip:
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
        print("using clip model: " + self.clip_model)

        if self.clip_cpu == False:
            self.clipDevice = self.vqganDevice
            if jit == False:
                self.clipPerceptor = clip.load(self.clip_model, jit=jit, download_root="./clipModels/")[0].eval().requires_grad_(False).to(self.clipDevice)
            else:
                self.clipPerceptor = clip.load(self.clip_model, jit=jit, download_root="./clipModels/")[0].eval().to(self.clipDevice)    
        else:
            self.clipDevice = torch.device("cpu")
            self.clipPerceptor = clip.load(self.clip_model, "cpu", jit=jit)[0].eval().requires_grad_(False).to(self.clipDevice) 



        print("---  CLIP model loaded to " + str(self.clipDevice) +" ---")
        self.log_torch_mem()
        print("--- / CLIP model loaded ---")

        self.clipPerceptorInputResolution = self.clipPerceptor.visual.input_resolution



    def InitVQGAN(self):
        self.vqganDevice = torch.device(self.cuda_device)

        self.vqganGumbelEnabled = False
        config = OmegaConf.load(self.vqgan_config_path)

        print("---  VQGAN config " + str(self.vqgan_config_path))    
        print(yaml.dump(OmegaConf.to_container(config)))
        print("---  / VQGAN config " + str(self.vqgan_config_path))

        if config.model.target == 'taming.models.vqgan.VQModel':
            self.vqganModel = vqgan.VQModel(**config.model.params)
            self.vqganModel.eval().requires_grad_(False)
            self.vqganModel.init_from_ckpt(self.vqgan_checkpoint_path)
        elif config.model.target == 'taming.models.vqgan.GumbelVQ':
            self.vqganModel = vqgan.GumbelVQ(**config.model.params)
            self.vqganModel.eval().requires_grad_(False)
            self.vqganModel.init_from_ckpt(self.vqgan_checkpoint_path)
            self.vqganGumbelEnabled = True
        elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
            parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
            parent_model.eval().requires_grad_(False)
            parent_model.init_from_ckpt(self.vqgan_checkpoint_path)
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
    def WriteLogClipResults(self, genJob:GenerateJob.GenerationJob, imgout:torch.Tensor):
        pass



    ######################
    ### interactive generation steps and training
    ######################

    def ProcessJobFullProfile(self, genJob:GenerateJob.GenerationJob, trainCallback = None):
        with torch.autograd.profiler.profile(use_cuda=True, with_stack=True) as prof:
            self.ProcessJobFull(genJob, trainCallback)

        #group_by_stack_n=5
        print("=========== CPU SELF =================")
        print( prof.key_averages().table(sort_by="self_cpu_time_total"))
        print("=========== CUDA SELF =================")
        print( prof.key_averages().table(sort_by="self_cuda_time_total"))
        print("=========== STACKs    =================")
        print( prof.key_averages(group_by_stack_n=60).table(sort_by="self_cuda_time_total"))

    def ProcessJobFull(self, genJob:GenerateJob.GenerationJob, trainCallback = None):        
            moreWork = True
            with tqdm() as pbar:
                while moreWork:   
                    # Training time         
                    moreWork = self.ProcessJobStep(genJob, trainCallback )
                    pbar.update()


    # step a job, returns true if theres more processing left for it
    def ProcessJobStep(self, genJob:GenerateJob.GenerationJob, trainCallbackFunc = None) -> bool:       
        #image manipulations before training is called, such as the zoom effect
        genJob.OnPreTrain()

        # Training time
        img, lossAll, lossSum = self.train(genJob, genJob.currentIteration)

        if trainCallbackFunc != None:
            trainCallbackFunc(genJob, genJob.currentIteration, img, lossAll, lossSum)
        
        self.DefaultTrainCallback(genJob, genJob.currentIteration, img, lossAll, lossSum)
   
        genJob.currentIteration += 1
        
        # Ready to stop yet?
        if genJob.currentIteration == genJob.totalIterations:
            genJob.OnFinishGeneration()    
            return False           
        
        return True

    
    @torch.inference_mode()
    def DefaultTrainCallback(self, genJob:GenerateJob.GenerationJob, iteration:int, curImg, lossAll, lossSum):
        # stat updates and progress images
        if iteration % self.display_freq == 0:
            print("\n*************************************************")
            print(f'i: {iteration}, loss sum: {lossSum.item():g}')
            print("*************************************************")

            promptNum = 0
            lossLen = len(lossAll)
            if genJob.embededPrompts and lossLen <= len(genJob.embededPrompts):
                for loss in lossAll:            
                    print( "----> " + genJob.embededPrompts[promptNum].TextPrompt + " - loss: " + str( loss.item() ) )
                    promptNum += 1
            else:
                print("mismatch in prompt numbers and losses!")

            print(" ")

            if self.log_clip:
                self.WriteLogClipResults(genJob, curImg)
                print(" ")

            if self.log_mem:
                self.log_torch_mem()
                print(" ")

            print(" ")
            sys.stdout.flush()  

        if iteration % genJob.save_freq == 0 and iteration != 0:     
            if genJob.save_seq == True:
                genJob.savedImageCount = genJob.savedImageCount + 1                
            else:
                genJob.savedImageCount = iteration
                
            genJob.SaveImageTensor( curImg, str(genJob.savedImageCount).zfill(5))
                            
        if genJob.save_best == True:

            lossAvg = lossSum / len(lossAll)

            if genJob.bestErrorScore > lossAvg.item():
                print("saving image for best error: " + str(lossAvg.item()))
                genJob.bestErrorScore = lossAvg
                genJob.SaveImageTensor( curImg, "lowest_error_")


    def train(self, genJob:GenerateJob.GenerationJob, iteration:int):
        with torch.cuda.amp.autocast(self.use_mixed_precision):
            genJob.optimizer.zero_grad(set_to_none=True)
            
            synthedImage = self.synth(genJob.quantizedImage, genJob.vqganGumbelEnabled) 
            
            cutouts = genJob.GetCutouts(synthedImage)

            if self.clipDevice != self.vqganDevice:
                clipEncodedImage = self.clipPerceptor.encode_image(self.normalize(cutouts.to(self.clipDevice))).float()
            else:
                clipEncodedImage = self.clipPerceptor.encode_image(self.normalize(cutouts)).float()

            
            lossAll = genJob.GetCutoutResults(clipEncodedImage, iteration)

            # see if this squaring helps with multiple prompts
            lossSum:torch.Tensor = None

            if len( lossAll ) > 1:
                total:torch.Tensor = None
                for t in lossAll:
                    if total == None:
                        total = torch.square( t )
                    else:
                        total += torch.square( t )

                lossSum = total
            else:
                ret:Any = sum(lossAll)
                assert( isinstance(ret, torch.Tensor) )
                lossSum = ret


            if genJob.optimizer == "MADGRAD":
                genJob.loss_idx.append(lossSum.item())
                if iteration > 100: #use only 100 last looses to avg
                    avg_loss = sum(genJob.loss_idx[iteration-100:])/len(genJob.loss_idx[iteration-100:]) 
                else:
                    avg_loss = sum(genJob.loss_idx)/len(genJob.loss_idx)

                genJob.scheduler.step(avg_loss)
            
            if self.use_mixed_precision == False:
                lossSum.backward()
                genJob.optimizer.step()
            else:
                genJob.gradScaler.scale(lossSum).backward()
                genJob.gradScaler.step(genJob.optimizer)
                genJob.gradScaler.update()
            
            with torch.inference_mode():
                genJob.quantizedImage.copy_(genJob.quantizedImage.maximum(genJob.z_min).minimum(genJob.z_max))

            return synthedImage, lossAll, lossSum

