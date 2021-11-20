# my mods from:
#
# inital repo:
# https://github.com/nerdyrodent/VQGAN-CLIP
#
# MSE
# https://www.reddit.com/r/bigsleep/comments/onmz5r/mse_regulized_vqgan_clip/
# https://colab.research.google.com/drive/1gFn9u3oPOgsNzJWEFmdK-N9h_y65b8fj?usp=sharing#scrollTo=wSfISAhyPmyp
#
# MADGRAD implementation reference
# https://www.kaggle.com/yannnobrega/vqgan-clip-z-quantize-method


# torch-optimizer info
# https://pypi.org/project/torch-optimizer/
#

# Originally made by Katherine Crowson (https://github.com/crowsonkb, https://twitter.com/RiversHaveWings)
# The original BigGAN+CLIP method was by https://twitter.com/advadnoun


# TODOS:
# - output is no longer deterministic as of updating to pytorch 1.10 and whatever other libs updated with it

import sys
import os
import random
import numpy as np


# shut off tqdm log spam by uncommenting the below
# from tqdm import tqdm
# from functools import partialmethod
# tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)



# trying to cut down on the absurd mess of a single file ...
sys.path.append('src')

from src import cmdLineArgs
cmdLineArgs.init()

from src import makeCutouts
makeCutouts.use_mixed_precision = cmdLineArgs.args.use_mixed_precision

from src import imageUtils


#stuff im using from source instead of installs
# i want to run clip from source, not an install. I have clip in a dir alongside this project
# so i append the parent dir to the proj and we expect to find a folder named clip there
sys.path.append('..\\')
from CLIP import clip


# pip install taming-transformers doesn't work with Gumbel, but does not yet work with coco etc
# appending the path does work with Gumbel, but gives ModuleNotFoundError: No module named 'transformers' for coco etc
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





def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

def log_torch_mem():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print("total mem:       " + str(t))
    print("reserved mem:    " + str(r))
    print("allocated mem:   " + str(a))
    print("free mem:        " + str(f))

def build_filename_path( filename ):
    fullpath = cmdLineArgs.args.output_dir
    if fullpath[-1] != '\\' and fullpath[-1] != '/':
        fullpath += '/'
    fullpath += filename
    return fullpath

# Set the optimiser
def get_opt(opt_name, opt_lr):
    if opt_name == "Adam":
        opt = optim.Adam([quantizedImage], lr=opt_lr)	# LR=0.1 (Default)
    elif opt_name == "AdamW":
        opt = optim.AdamW([quantizedImage], lr=opt_lr)	
    elif opt_name == "Adagrad":
        opt = optim.Adagrad([quantizedImage], lr=opt_lr)	
    elif opt_name == "Adamax":
        opt = optim.Adamax([quantizedImage], lr=opt_lr)	
    elif opt_name == "DiffGrad":
        opt = torch_optimizer.DiffGrad([quantizedImage], lr=opt_lr, eps=1e-9, weight_decay=1e-9) # NR: Playing for reasons
    elif opt_name == "AdamP":
        opt = torch_optimizer.AdamP([quantizedImage], lr=opt_lr)		    	    
    elif opt_name == "RMSprop":
        opt = optim.RMSprop([quantizedImage], lr=opt_lr)
    elif opt_name == "MADGRAD":
        opt = torch_optimizer.MADGRAD([quantizedImage], lr=opt_lr)      
    else:
        print("Unknown optimiser. Are choices broken?")
        opt = optim.Adam([quantizedImage], lr=opt_lr)
    return opt




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


#@torch.inference_mode() -> gets error on backward loss that tensor has grad disabled
def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)


class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))

    @autocast(enabled=cmdLineArgs.args.use_mixed_precision)
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


def load_vqgan_model(config_path, checkpoint_path):
    global gumbel
    gumbel = False
    config = OmegaConf.load(config_path)

    print("---  VQGAN config " + str(config_path))    
    print(yaml.dump(OmegaConf.to_container(config)))
    print("---  / VQGAN config " + str(config_path))

    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.vqgan.GumbelVQ':
        model = vqgan.GumbelVQ(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
        gumbel = True
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model

# Vector quantize
def synth(z):
    if gumbel:
        z_q = vector_quantize(z.movedim(1, 3), vqganModel.quantize.embed.weight).movedim(3, 1)
    else:
        z_q = vector_quantize(z.movedim(1, 3), vqganModel.quantize.embedding.weight).movedim(3, 1)
    return makeCutouts.clamp_with_grad(vqganModel.decode(z_q).add(1).div(2), 0, 1)



## calleed during training / end of training to log info, save files, etc.
@torch.inference_mode()
def WriteLogClipResults(imgout):
    #image, class_id = cifar100[3637]

    #out = synth(z) 
    img = normalize(make_cutouts(imgout))

    if cmdLineArgs.args.log_clip_oneshot:
        #one shot identification
        image_features = clipPerceptor.encode_image(img).float()

        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(clipDevice)
        
        text_features = clipPerceptor.encode_text(text_inputs).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(5)

        # Print the result
        print("\nOne-shot predictions:\n")
        for value, index in zip(values, indices):
            print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")

    if cmdLineArgs.args.log_clip:
        # prompt matching percentages
        textins = []
        promptPartStrs = []
        if cmdLineArgs.args.prompts:
            for prompt in cmdLineArgs.args.prompts:
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

        #text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(clipDevice)
        text_inputs = torch.cat(textins).to(clipDevice)
        
        image_features = clipPerceptor.encode_image(img).float()
        text_features = clipPerceptor.encode_text(text_inputs).float()
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


@torch.inference_mode()
def checkin(i, losses, out):
    print("\n*************************************************")
    print(f'i: {i}, loss sum: {sum(losses).item():g}')
    print("*************************************************")

    promptNum = 0
    if cmdLineArgs.args.prompts:
        for loss in losses:
            print( "----> " + cmdLineArgs.args.prompts[promptNum] + " - loss: " + str(loss.item()) )
            promptNum += 1

    print(" ")

    if cmdLineArgs.args.log_clip:
        WriteLogClipResults(out)
        print(" ")

    info = PngImagePlugin.PngInfo()
    info.add_text('comment', f'{cmdLineArgs.args.prompts}')
    TF.to_pil_image(out[0].cpu()).save( build_filename_path( str(i).zfill(5) + cmdLineArgs.args.output ), pnginfo=info)
    
    if cmdLineArgs.args.log_mem:
        log_torch_mem()
        print(" ")

    print(" ")
    sys.stdout.flush()

    #gc.collect()


def ascend_txt(out):
    global iteration
    with torch.cuda.amp.autocast(cmdLineArgs.args.use_mixed_precision):

        cutouts = make_cutouts(out)

        if clipDevice != vqganDevice:
            iii = clipPerceptor.encode_image(normalize(cutouts.to(clipDevice))).float()
        else:
            iii = clipPerceptor.encode_image(normalize(cutouts)).float()

        result = []        

        if cmdLineArgs.args.init_weight:
            # result.append(F.mse_loss(z, z_orig) * args.init_weight / 2)
            result.append(F.mse_loss(quantizedImage, torch.zeros_like(z_orig)) * ((1/torch.tensor(iteration*2 + 1))*cmdLineArgs.args.init_weight) / 2)

        for prompt in pMs:
            result.append(prompt(iii))
        
        return result # return loss



bestErrorScore = 99999
def train(i):
    global loss_idx
    global bestErrorScore

    with torch.cuda.amp.autocast(cmdLineArgs.args.use_mixed_precision):
        opt.zero_grad(set_to_none=True)
        
        out = synth(quantizedImage) 
        
        lossAll = ascend_txt(out)

        if i % cmdLineArgs.args.display_freq == 0:
            checkin(i, lossAll, out)  

        if i % cmdLineArgs.args.save_freq == 0:          
            info = PngImagePlugin.PngInfo()
            info.add_text('comment', f'{cmdLineArgs.args.prompts}')
            TF.to_pil_image(out[0].cpu()).save( build_filename_path( str(i).zfill(5) + cmdLineArgs.args.output) , pnginfo=info)
            
        loss = sum(lossAll)
        lossAvg = loss / len(lossAll)

        with torch.inference_mode():
            if cmdLineArgs.args.save_best == True and bestErrorScore > lossAvg.item():
                print("saving image for best error: " + str(lossAvg.item()))
                bestErrorScore = lossAvg
                info = PngImagePlugin.PngInfo()
                info.add_text('comment', f'{cmdLineArgs.args.prompts}')
                TF.to_pil_image(out[0].cpu()).save( build_filename_path( "lowest_error_" + cmdLineArgs.args.output), pnginfo=info)

        if cmdLineArgs.args.optimiser == "MADGRAD":
            loss_idx.append(loss.item())
            if i > 100: #use only 100 last looses to avg
                avg_loss = sum(loss_idx[i-100:])/len(loss_idx[i-100:]) 
            else:
                avg_loss = sum(loss_idx)/len(loss_idx)

            scheduler.step(avg_loss)
        
        if cmdLineArgs.args.use_mixed_precision == False:
            loss.backward()
            opt.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        
        #with torch.no_grad():
        with torch.inference_mode():
            quantizedImage.copy_(quantizedImage.maximum(z_min).minimum(z_max))









###########################################################
# start actually doing stuff here.... process cmd line args
# #########################################################
print("Args: " + str(cmdLineArgs.args) )
print("Using pyTorch: " + str( torch.__version__) )
print( "Using mixed precision: " + str(cmdLineArgs.args.use_mixed_precision) )  

if cmdLineArgs.args.seed is None:
    seed = torch.seed()
else:
    seed = cmdLineArgs.args.seed  

print('Using seed:', seed)
seed_torch(seed)

if cmdLineArgs.args.deterministic >= 2:
    print("Determinism at max: forcing a lot of things so this will work, no augs, non-pooling cut method, bad resampling")
    cmdLineArgs.args.augments = "none"
    cmdLineArgs.args.cut_method = "original"

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
elif cmdLineArgs.args.deterministic == 1:
    print("Determinism at medium: cudnn determinism and benchmark disabled")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # NR: True is a bit faster, but can lead to OOM. False is more deterministic.
else:
    print("Determinism at minimum: cudnn benchmark on")
    torch.backends.cudnn.benchmark = True #apparently slightly faster, but less deterministic


os.makedirs(os.path.dirname(cmdLineArgs.args.output_dir), exist_ok=True)

if cmdLineArgs.args.log_clip:
    print("logging clip probabilities at end, loading vocab stuff")
    cifar100 = CIFAR100(root=".", download=True, train=False)

if not cmdLineArgs.args.prompts and not cmdLineArgs.args.image_prompts:
    cmdLineArgs.args.prompts = "A cute, smiling, Nerdy Rodent"

if not cmdLineArgs.args.augments:
   cmdLineArgs.args.augments = [['Af', 'Pe', 'Ji', 'Er']]
elif cmdLineArgs.args.augments == 'None':
    print("Augments set to none")
    cmdLineArgs.args.augments = []

if cmdLineArgs.args.use_mixed_precision==True:
    print("cant use augments in mixed precision mode yet...")
    cmdLineArgs.args.augments = []

# Split text prompts using the pipe character (weights are split later)
if cmdLineArgs.args.prompts:
    # For stories, there will be many phrases
    story_phrases = [phrase.strip() for phrase in cmdLineArgs.args.prompts.split("^")]
    
    # Make a list of all phrases
    all_phrases = []
    for phrase in story_phrases:
        all_phrases.append(phrase.split("|"))
    
    # First phrase
    cmdLineArgs.args.prompts = all_phrases[0]
    
# Split target images using the pipe character (weights are split later)
if cmdLineArgs.args.image_prompts:
    cmdLineArgs.args.image_prompts = cmdLineArgs.args.image_prompts.split("|")
    cmdLineArgs.args.image_prompts = [image.strip() for image in cmdLineArgs.args.image_prompts]

# Fallback to CPU if CUDA is not found and make sure GPU video rendering is also disabled
# NB. May not work for AMD cards?
if not cmdLineArgs.args.cuda_device == 'cpu' and not torch.cuda.is_available():
    cmdLineArgs.args.cuda_device = 'cpu'
    print("Warning: No GPU found! Using the CPU instead. The iterations will be slow.")
    print("Perhaps CUDA/ROCm or the right pytorch version is not properly installed?")

# Do it
vqganDevice = torch.device(cmdLineArgs.args.cuda_device)
vqganModel = load_vqgan_model(cmdLineArgs.args.vqgan_config, cmdLineArgs.args.vqgan_checkpoint).to(vqganDevice)

print("---  VQGAN model loaded ---")
log_torch_mem()
print("--- / VQGAN model loaded ---")

jit = False
try:
    # try here, since using nightly build of pytorch has a version scheme like dev23723h
    if [int(n) for n in torch.__version__.split(".")] < [1, 8, 1]:
        jit = True
except:
    jit = False

print( "available clip models: " + str(clip.available_models() ))
print("CLIP jit: " + str(jit))
print("using clip model: " + cmdLineArgs.args.clip_model)

if cmdLineArgs.args.clip_cpu == False:
    clipDevice = vqganDevice
    # hmm, updating to latest pytorch complains about .requires_grad being invalid? this works without jit
    if jit == False:
        clipPerceptor = clip.load(cmdLineArgs.args.clip_model, jit=jit, download_root="./clipModels/")[0].eval().requires_grad_(False).to(clipDevice)
    else:
        clipPerceptor = clip.load(cmdLineArgs.args.clip_model, jit=jit, download_root="./clipModels/")[0].eval().to(clipDevice)    
else:
    clipDevice = torch.device("cpu")
    clipPerceptor = clip.load(cmdLineArgs.args.clip_model, "cpu", jit=jit)[0].eval().requires_grad_(False).to(clipDevice) 



print("---  CLIP model loaded to " + str(clipDevice) +" ---")
log_torch_mem()
print("--- / CLIP model loaded ---")

if cmdLineArgs.args.anomaly_checker:
    torch.autograd.set_detect_anomaly(True)


# clock=deepcopy(perceptor.visual.positional_embedding.data)
# perceptor.visual.positional_embedding.data = clock/clock.max()
# perceptor.visual.positional_embedding.data=clamp_with_grad(clock,0,1)

clipPerceptorInputResolution = clipPerceptor.visual.input_resolution
vqganNumResolutionsF = 2**(vqganModel.decoder.num_resolutions - 1)
toksX, toksY = cmdLineArgs.args.size[0] // vqganNumResolutionsF, cmdLineArgs.args.size[1] // vqganNumResolutionsF
sideX, sideY = toksX * vqganNumResolutionsF, toksY * vqganNumResolutionsF

print("vqgan input resolutions: " + str(vqganModel.decoder.num_resolutions));
print("cliperceptor input_res (aka cut size): " + str(clipPerceptorInputResolution) + " and whatever f is supposed to be: " + str(vqganNumResolutionsF))
print("Toks X,Y: " + str(toksX) + ", " + str(toksY) + "      SizeX,Y: " + str(sideX) + ", " + str(sideY))
# from default run:
# vqgan input resolutions: 5
# cliperceptor input_res: 224 and whatever f is supposed to be: 16
# Toks X,Y: 75, 75      SizeX,Y: 1200 1200

# Cutout class options:
# 'squish', 'latest','original','updated' or 'updatedpooling'
if cmdLineArgs.args.cut_method == 'latest':
    make_cutouts = makeCutouts.MakeCutouts(clipPerceptorInputResolution, cmdLineArgs.args.cutn, cut_pow=cmdLineArgs.args.cut_pow, augments=cmdLineArgs.args.augments)
elif cmdLineArgs.args.cut_method == 'squish':
    cutSize = cmdLineArgs.args.cut_size
    if cutSize[0] == 0:
        cutSize[0] = clipPerceptorInputResolution

    if cutSize[1] == 0:
        cutSize[1] = clipPerceptorInputResolution        

    print("Squish Cutouts using: cutSize " + str(cutSize))

    # pooling requires proper matching sizes for now
    if clipPerceptorInputResolution != cutSize or clipPerceptorInputResolution != cutSize[0] or clipPerceptorInputResolution != cutSize[1]:
        make_cutouts = makeCutouts.MakeCutoutsSquish(clipPerceptorInputResolution, cutSize[0], cutSize[1], cmdLineArgs.args.cutn, cut_pow=cmdLineArgs.args.cut_pow, use_pool=True, augments=cmdLineArgs.args.augments)
    else:
        make_cutouts = makeCutouts.MakeCutoutsSquish(clipPerceptorInputResolution, cutSize[0], cutSize[1], cmdLineArgs.args.cutn, cut_pow=cmdLineArgs.args.cut_pow, use_pool=True, augments=cmdLineArgs.args.augments)

elif cmdLineArgs.args.cut_method == 'original':
    make_cutouts = makeCutouts.MakeCutoutsOrig(clipPerceptorInputResolution, cmdLineArgs.args.cutn, cut_pow=cmdLineArgs.args.cut_pow, augments=cmdLineArgs.args.augments)
elif cmdLineArgs.args.cut_method == 'updated':
    make_cutouts = makeCutouts.MakeCutoutsUpdate(clipPerceptorInputResolution, cmdLineArgs.args.cutn, cut_pow=cmdLineArgs.args.cut_pow, augments=cmdLineArgs.args.augments)
elif cmdLineArgs.args.cut_method == 'nrupdated':
    make_cutouts = makeCutouts.MakeCutoutsNRUpdate(clipPerceptorInputResolution, cmdLineArgs.args.cutn, cut_pow=cmdLineArgs.args.cut_pow, augments=cmdLineArgs.args.augments)
else:
    make_cutouts = makeCutouts.MakeCutoutsPoolingUpdate(clipPerceptorInputResolution, cmdLineArgs.args.cutn, cut_pow=cmdLineArgs.args.cut_pow, augments=cmdLineArgs.args.augments)    



# Gumbel or not?
if gumbel:
    e_dim = 256
    n_toks = vqganModel.quantize.n_embed
    z_min = vqganModel.quantize.embed.weight.min(dim=0).values[None, :, None, None]
    z_max = vqganModel.quantize.embed.weight.max(dim=0).values[None, :, None, None]
else:
    e_dim = vqganModel.quantize.e_dim
    n_toks = vqganModel.quantize.n_e
    z_min = vqganModel.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = vqganModel.quantize.embedding.weight.max(dim=0).values[None, :, None, None]


if cmdLineArgs.args.init_image:
    if 'http' in cmdLineArgs.args.init_image:
        img = Image.open(urlopen(cmdLineArgs.args.init_image))
    else:
        img = Image.open(cmdLineArgs.args.init_image)
        pil_image = img.convert('RGB')
        pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
        pil_tensor = TF.to_tensor(pil_image)
        quantizedImage, *_ = vqganModel.encode(pil_tensor.to(vqganDevice).unsqueeze(0) * 2 - 1)
elif cmdLineArgs.args.init_noise == 'pixels':
    img = imageUtils.random_noise_image(cmdLineArgs.args.size[0], cmdLineArgs.args.size[1])    
    pil_image = img.convert('RGB')
    pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
    pil_tensor = TF.to_tensor(pil_image)
    quantizedImage, *_ = vqganModel.encode(pil_tensor.to(vqganDevice).unsqueeze(0) * 2 - 1)
elif cmdLineArgs.args.init_noise == 'gradient':
    img = imageUtils.random_gradient_image(cmdLineArgs.args.size[0], cmdLineArgs.args.size[1])
    pil_image = img.convert('RGB')
    pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
    pil_tensor = TF.to_tensor(pil_image)
    quantizedImage, *_ = vqganModel.encode(pil_tensor.to(vqganDevice).unsqueeze(0) * 2 - 1)
else:
    one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=vqganDevice), n_toks).float()
    # z = one_hot @ vqganModel.quantize.embedding.weight
    if gumbel:
        quantizedImage = one_hot @ vqganModel.quantize.embed.weight
    else:
        quantizedImage = one_hot @ vqganModel.quantize.embedding.weight

    quantizedImage = quantizedImage.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2) 
    #z = torch.rand_like(z)*2						# NR: check


z_orig = quantizedImage.clone()
z_orig.requires_grad_(False)

quantizedImage.requires_grad_(True)

# write out the input noise...
out = synth(z_orig)
info = PngImagePlugin.PngInfo()
info.add_text('comment', f'{cmdLineArgs.args.prompts}')
TF.to_pil_image(out[0].cpu()).save( build_filename_path( str(0).zfill(5) + '_seed_' + cmdLineArgs.args.output ), pnginfo=info)

pMs = []
normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                  std=[0.26862954, 0.26130258, 0.27577711])

# From imagenet - Which is better?
#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])

# CLIP tokenize/encode   
if cmdLineArgs.args.prompts:
    for prompt in cmdLineArgs.args.prompts:
        txt, weight, stop = split_prompt(prompt)
        embed = clipPerceptor.encode_text(clip.tokenize(txt).to(clipDevice)).float()
        pMs.append(Prompt(embed, weight, stop).to(clipDevice))

for prompt in cmdLineArgs.args.image_prompts:
    path, weight, stop = split_prompt(prompt)
    img = Image.open(path)
    pil_image = img.convert('RGB')
    img = imageUtils.resize_image(pil_image, (sideX, sideY))
    batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(clipDevice))
    embed = clipPerceptor.encode_image(normalize(batch)).float()
    pMs.append(Prompt(embed, weight, stop).to(clipDevice))

for seed, weight in zip(cmdLineArgs.args.noise_prompt_seeds, cmdLineArgs.args.noise_prompt_weights):
    gen = torch.Generator().manual_seed(seed)
    embed = torch.empty([1, clipPerceptor.visual.output_dim]).normal_(generator=gen)
    pMs.append(Prompt(embed, weight).to(clipDevice))

opt = get_opt(cmdLineArgs.args.optimiser,cmdLineArgs. args.step_size)

if cmdLineArgs.args.optimiser == "MADGRAD":
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.999, patience=0)  

# Output for the user
print('Using vqgandevice:', vqganDevice)
print('Using clipdevice:', clipDevice)
print('Optimising using:', cmdLineArgs.args.optimiser)

if cmdLineArgs.args.prompts:
    print('Using text prompts:', cmdLineArgs.args.prompts)  
if cmdLineArgs.args.image_prompts:
    print('Using image prompts:', cmdLineArgs.args.image_prompts)
if cmdLineArgs.args.init_image:
    print('Using initial image:', cmdLineArgs.args.init_image)
if cmdLineArgs.args.noise_prompt_weights:
    print('Noise prompt weights:', cmdLineArgs.args.noise_prompt_weights)    

loss_idx = []
iteration = 0 # Iteration counter
phraseCounter = 1 # Phrase counter

# Messing with learning rate / optimisers
#variable_lr = args.step_size
#optimiser_list = [['Adam',0.075],['AdamW',0.125],['Adagrad',0.2],['Adamax',0.125],['DiffGrad',0.075],['RAdam',0.125],['RMSprop',0.02]]

# Creates a GradScaler once at the beginning of training.
scaler = GradScaler()

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
                    if phraseCounter >= len(all_phrases):
                        phraseCounter = 0
                    
                    pMs = []
                    cmdLineArgs.args.prompts = all_phrases[phraseCounter]

                    # Show user we're changing prompt                                
                    print(cmdLineArgs.args.prompts)
                    
                    for prompt in cmdLineArgs.args.prompts:
                        txt, weight, stop = split_prompt(prompt)
                        embed = clipPerceptor.encode_text(clip.tokenize(txt).to(clipDevice)).float()
                        pMs.append(Prompt(embed, weight, stop).to(clipDevice))


                    phraseCounter += 1
            

            # Training time
            train(iteration)
           
            
            # Ready to stop yet?
            if iteration == cmdLineArgs.args.max_iterations:
                # Save final image
                out = synth(quantizedImage)                                    
                img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:,:,:]
                img = np.transpose(img, (1, 2, 0))
                imageio.imwrite(build_filename_path(cmdLineArgs.args.output), np.array(img))                

                if cmdLineArgs.args.log_clip:    
                    # write one for the console
                    WriteLogClipResults(out)
                	#write once to a file for easy grabbing outside of this script                
                    text_file = open(build_filename_path( cmdLineArgs.args.output + ".txt"), "w")
                    sys.stdout = text_file
                    WriteLogClipResults(out)
                    sys.stdout = sys.stdout 
                    text_file.close()

                break



            iteration += 1
            pbar.update()

except KeyboardInterrupt:
    pass


