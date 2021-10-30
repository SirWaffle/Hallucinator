# my mods from:
#
# inital repo:
# https://github.com/nerdyrodent/VQGAN-CLIP
#
# MSE
# https://www.reddit.com/r/bigsleep/comments/onmz5r/mse_regulized_vqgan_clip/
# https://colab.research.google.com/drive/1gFn9u3oPOgsNzJWEFmdK-N9h_y65b8fj?usp=sharing#scrollTo=wSfISAhyPmyp
#
# MADGRAD
# https://github.com/facebookresearch/madgrad
# https://www.kaggle.com/yannnobrega/vqgan-clip-z-quantize-method


# torch-optimizer info
# https://pypi.org/project/torch-optimizer/
#

# Originally made by Katherine Crowson (https://github.com/crowsonkb, https://twitter.com/RiversHaveWings)
# The original BigGAN+CLIP method was by https://twitter.com/advadnoun

# trying to cut down on the absurd mess of a single file
import cmdLineArgs
cmdLineArgs.init()


import makeCutouts


# back to original list of imports
from madgrad import MADGRAD

import random
# from email.policy import default
from urllib.request import urlopen
from tqdm import tqdm
import sys
import os

# pip install taming-transformers doesn't work with Gumbel, but does not yet work with coco etc
# appending the path does work with Gumbel, but gives ModuleNotFoundError: No module named 'transformers' for coco etc
sys.path.append('taming-transformers')

from omegaconf import OmegaConf
from taming.models import cond_transformer, vqgan
#import taming.modules 

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
torch.backends.cudnn.benchmark = False		# NR: True is a bit faster, but can lead to OOM. False is more deterministic.
#torch.use_deterministic_algorithms(True)	# NR: grid_sampler_2d_backward_cuda does not have a deterministic implementation

from torch_optimizer import DiffGrad, AdamP

from CLIP import clip
import kornia.augmentation as K
import numpy as np
import imageio

import random

from PIL import ImageFile, Image, PngImagePlugin, ImageChops
ImageFile.LOAD_TRUNCATED_IMAGES = True

from subprocess import Popen, PIPE
import re

from torchvision.datasets import CIFAR100

# Supress warnings
import warnings
warnings.filterwarnings('ignore')






def log_torch_mem():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print("total mem:       " + str(t))
    print("reserved mem:    " + str(r))
    print("allocated mem:   " + str(a))
    print("free mem:        " + str(f))


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.





# start actually doing stuff here.... process cmd line args    


if cmdLineArgs.args.log_clip:
    print("logging clip probabilities at end, loading vocab stuff")
    cifar100 = CIFAR100(root=".", download=True, train=False)

print( "Using mixed precision: " + str(cmdLineArgs.args.use_mixed_precision) )  

if not cmdLineArgs.args.prompts and not cmdLineArgs.args.image_prompts:
    cmdLineArgs.args.prompts = "A cute, smiling, Nerdy Rodent"

if cmdLineArgs.args.cudnn_determinism:
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False

if not cmdLineArgs.args.augments:
   cmdLineArgs.args.augments = [['Af', 'Pe', 'Ji', 'Er']]

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

if cmdLineArgs.args.make_video and cmdLineArgs.args.make_zoom_video:
    print("Warning: Make video and make zoom video are mutually exclusive.")
    cmdLineArgs.args.make_video = False
    
# Make video steps directory
if cmdLineArgs.args.make_video or cmdLineArgs.args.make_zoom_video:
    if not os.path.exists('steps'):
        os.mkdir('steps')

# Fallback to CPU if CUDA is not found and make sure GPU video rendering is also disabled
# NB. May not work for AMD cards?
if not cmdLineArgs.args.cuda_device == 'cpu' and not torch.cuda.is_available():
    cmdLineArgs.args.cuda_device = 'cpu'
    cmdLineArgs.args.video_fps = 0
    print("Warning: No GPU found! Using the CPU instead. The iterations will be slow.")
    print("Perhaps CUDA/ROCm or the right pytorch version is not properly installed?")

# If a video_style_dir has been, then create a list of all the images
if cmdLineArgs.args.video_style_dir:
    print("Locating video frames...")
    video_frame_list = []
    for entry in os.scandir(cmdLineArgs.args.video_style_dir):
        if (entry.path.endswith(".jpg")
                or entry.path.endswith(".png")) and entry.is_file():
            video_frame_list.append(entry.path)

    # Reset a few options - same filename, different directory
    if not os.path.exists('steps'):
        os.mkdir('steps')

    cmdLineArgs.args.init_image = video_frame_list[0]
    filename = os.path.basename(args.init_image)
    cwd = os.getcwd()
    cmdLineArgs.args.output = os.path.join(cwd, "steps", filename)
    num_video_frames = len(video_frame_list) # for video styling


# For zoom video
def zoom_at(img, x, y, zoom):
    w, h = img.size
    zoom2 = zoom * 2
    img = img.crop((x - w / zoom2, y - h / zoom2, 
                    x + w / zoom2, y + h / zoom2))
    return img.resize((w, h), Image.LANCZOS)


# NR: Testing with different intital images
def random_noise_image(w,h):
    print('generating random noise image')
    random_image = Image.fromarray(nn.random.randint(0,255,(w,h,3),dtype=np.dtype('uint8')))
    return random_image


# create initial gradient image
def gradient_2d(start, stop, width, height, is_horizontal):
    print('generating gradient2d random noise image')
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T


def gradient_3d(width, height, start_list, stop_list, is_horizontal_list):
    print('generating gradient3d random noise image')
    result = np.zeros((height, width, len(start_list)), dtype=float)

    for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):
        result[:, :, i] = gradient_2d(start, stop, width, height, is_horizontal)

    return result

    
def random_gradient_image(w,h):
    print('generating random gradient noise image')
    array = gradient_3d(w, h, (0, 0, np.random.randint(0,255)), (np.random.randint(1,255), np.random.randint(2,255), np.random.randint(3,128)), (True, False, False))
    random_image = Image.fromarray(np.uint8(array))
    return random_image


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


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None

clamp_with_grad = ClampWithGrad.apply


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
    return clamp_with_grad(vqganModel.decode(z_q).add(1).div(2), 0, 1)
    
def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)


# Do it
vqganDevice = torch.device(cmdLineArgs.args.cuda_device)
vqganModel = load_vqgan_model(cmdLineArgs.args.vqgan_config, cmdLineArgs.args.vqgan_checkpoint).to(vqganDevice)

print("---  VQGAN model loaded ---")
log_torch_mem()
print("--- / VQGAN model loaded ---")


jit = True if float(torch.__version__[:3]) < 1.8 else False


if cmdLineArgs.args.clip_cpu == False:
    clipDevice = vqganDevice
    clipPerceptor = clip.load(cmdLineArgs.args.clip_model, jit=jit)[0].eval().requires_grad_(False).to(clipDevice)       
else:
    clipDevice = torch.device("cpu")
    clipPerceptor = clip.load(cmdLineArgs.args.clip_model, "cpu", jit=jit)[0].eval().requires_grad_(False).to(clipDevice) 



print("---  CLIP model loaded to " + str(clipDevice) +" ---")
log_torch_mem()
print("--- / CLIP model loaded ---")

if cmdLineArgs.args.anomaly_checker:
    torch.autograd.set_detect_anomaly(True)


if cmdLineArgs.args.seed is None:
    seed = torch.seed()
else:
    seed = cmdLineArgs.args.seed  

print('Using seed:', seed)
seed_torch(seed)


# clock=deepcopy(perceptor.visual.positional_embedding.data)
# perceptor.visual.positional_embedding.data = clock/clock.max()
# perceptor.visual.positional_embedding.data=clamp_with_grad(clock,0,1)

cut_size = clipPerceptor.visual.input_resolution
f = 2**(vqganModel.decoder.num_resolutions - 1)

# Cutout class options:
# 'latest','original','updated' or 'updatedpooling'
if cmdLineArgs.args.cut_method == 'latest':
    make_cutouts = makeCutouts.MakeCutouts(cut_size, cmdLineArgs.args.cutn, cut_pow=cmdLineArgs.args.cut_pow)
elif cmdLineArgs.args.cut_method == 'original':
    make_cutouts = makeCutouts.MakeCutoutsOrig(cut_size, cmdLineArgs.args.cutn, cut_pow=cmdLineArgs.args.cut_pow)
elif cmdLineArgs.args.cut_method == 'updated':
    make_cutouts = makeCutouts.MakeCutoutsUpdate(cut_size, cmdLineArgs.args.cutn, cut_pow=cmdLineArgs.args.cut_pow)
elif cmdLineArgs.args.cut_method == 'nrupdated':
    make_cutouts = makeCutouts.MakeCutoutsNRUpdate(cut_size, cmdLineArgs.args.cutn, cut_pow=cmdLineArgs.args.cut_pow)
else:
    make_cutouts = makeCutouts.MakeCutoutsPoolingUpdate(cut_size, cmdLineArgs.args.cutn, cut_pow=cmdLineArgs.args.cut_pow)    

toksX, toksY = cmdLineArgs.args.size[0] // f, cmdLineArgs.args.size[1] // f
sideX, sideY = toksX * f, toksY * f

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
        z, *_ = vqganModel.encode(pil_tensor.to(vqganDevice).unsqueeze(0) * 2 - 1)
elif cmdLineArgs.args.init_noise == 'pixels':
    img = random_noise_image(cmdLineArgs.args.size[0], cmdLineArgs.args.size[1])    
    pil_image = img.convert('RGB')
    pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
    pil_tensor = TF.to_tensor(pil_image)
    z, *_ = vqganModel.encode(pil_tensor.to(vqganDevice).unsqueeze(0) * 2 - 1)
elif cmdLineArgs.args.init_noise == 'gradient':
    img = random_gradient_image(cmdLineArgs.args.size[0], cmdLineArgs.args.size[1])
    pil_image = img.convert('RGB')
    pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
    pil_tensor = TF.to_tensor(pil_image)
    z, *_ = vqganModel.encode(pil_tensor.to(vqganDevice).unsqueeze(0) * 2 - 1)
else:
    one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=vqganDevice), n_toks).float()
    # z = one_hot @ vqganModel.quantize.embedding.weight
    if gumbel:
        z = one_hot @ vqganModel.quantize.embed.weight
    else:
        z = one_hot @ vqganModel.quantize.embedding.weight

    z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2) 
    #z = torch.rand_like(z)*2						# NR: check

# attempt to write out the input noise...
out = synth(z)
info = PngImagePlugin.PngInfo()
info.add_text('comment', f'{cmdLineArgs.args.prompts}')
TF.to_pil_image(out[0].cpu()).save( str(0).zfill(5) + '_seed_' + cmdLineArgs.args.output, pnginfo=info)

 
z_orig = z.clone()

z.requires_grad_(True)

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
    img = resize_image(pil_image, (sideX, sideY))
    batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(clipDevice))
    embed = clipPerceptor.encode_image(normalize(batch)).float()
    pMs.append(Prompt(embed, weight, stop).to(clipDevice))

for seed, weight in zip(cmdLineArgs.args.noise_prompt_seeds, cmdLineArgs.args.noise_prompt_weights):
    gen = torch.Generator().manual_seed(seed)
    embed = torch.empty([1, clipPerceptor.visual.output_dim]).normal_(generator=gen)
    pMs.append(Prompt(embed, weight).to(clipDevice))


# Set the optimiser
def get_opt(opt_name, opt_lr):
    if opt_name == "Adam":
        opt = optim.Adam([z], lr=opt_lr)	# LR=0.1 (Default)
    elif opt_name == "AdamW":
        opt = optim.AdamW([z], lr=opt_lr)	
    elif opt_name == "Adagrad":
        opt = optim.Adagrad([z], lr=opt_lr)	
    elif opt_name == "Adamax":
        opt = optim.Adamax([z], lr=opt_lr)	
    elif opt_name == "DiffGrad":
        opt = DiffGrad([z], lr=opt_lr, eps=1e-9, weight_decay=1e-9) # NR: Playing for reasons
    elif opt_name == "AdamP":
        opt = AdamP([z], lr=opt_lr)		    	    
    elif opt_name == "RMSprop":
        opt = optim.RMSprop([z], lr=opt_lr)
    elif opt_name == "MADGRAD":
        opt = MADGRAD([z], lr=args.step_size)             
    else:
        print("Unknown optimiser. Are choices broken?")
        opt = optim.Adam([z], lr=opt_lr)
    return opt

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

def WriteLogClipResults(imgout):
    #image, class_id = cifar100[3637]

    #out = synth(z) 
    img = normalize(make_cutouts(imgout))

    if cmdLineArgs.args.log_clip_oneshot:
        #one shot identification
        with torch.no_grad():        
            image_features = clipPerceptor.encode_image(img).float()

        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(clipDevice)
        
        with torch.no_grad():
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
        
        with torch.no_grad():
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
    #losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
    #tqdm.write(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')

    print("\n*************************************************")
    print(f'i: {i}, loss sum: {sum(losses).item():g}')
    print("*************************************************")

    promptNum = 0
    for loss in losses:
        print( "----> " + cmdLineArgs.args.prompts[promptNum] + " - loss: " + str(loss.item()) )
        promptNum += 1

    print(" ")

    if cmdLineArgs.args.log_clip:
        WriteLogClipResults(out)
        print(" ")

    info = PngImagePlugin.PngInfo()
    info.add_text('comment', f'{cmdLineArgs.args.prompts}')
    TF.to_pil_image(out[0].cpu()).save( str(i).zfill(5) + cmdLineArgs.args.output, pnginfo=info)
    
    if cmdLineArgs.args.log_mem:
        log_torch_mem()
        print(" ")

    print(" ")

    #torch.cuda.empty_cache()
    #gc.collect()


def ascend_txt(out):

    global i
    with torch.cuda.amp.autocast(cmdLineArgs.args.use_mixed_precision):

        if clipDevice != vqganDevice:
            iii = clipPerceptor.encode_image(normalize(make_cutouts(out).to(clipDevice))).float()
        else:
            iii = clipPerceptor.encode_image(normalize(make_cutouts(out))).float()

        
        result = []

        if cmdLineArgs.args.init_weight:
            # result.append(F.mse_loss(z, z_orig) * args.init_weight / 2)
            result.append(F.mse_loss(z, torch.zeros_like(z_orig)) * ((1/torch.tensor(i*2 + 1))*cmdLineArgs.args.init_weight) / 2)

        for prompt in pMs:
            result.append(prompt(iii))
        
        if cmdLineArgs.args.make_video:    
            img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:,:,:]
            img = np.transpose(img, (1, 2, 0))
            imageio.imwrite('./steps/' + str(i) + '.png', np.array(img))

        return result # return loss


bestErrorScore = 99999

def train(i):
    global loss_idx
    global bestErrorScore

    with torch.cuda.amp.autocast(cmdLineArgs.args.use_mixed_precision):
        opt.zero_grad(set_to_none=True)
        
        out = synth(z) 
        
        lossAll = ascend_txt(out)
        
        if i % cmdLineArgs.args.display_freq == 0:
            checkin(i, lossAll, out)  

        if i % cmdLineArgs.args.save_freq == 0:          
            info = PngImagePlugin.PngInfo()
            info.add_text('comment', f'{cmdLineArgs.args.prompts}')
            TF.to_pil_image(out[0].cpu()).save( str(i).zfill(5) + cmdLineArgs.args.output, pnginfo=info)
            
        loss = sum(lossAll)
        lossAvg = loss / len(lossAll)

        if cmdLineArgs.args.save_best == True and bestErrorScore > lossAvg.item():
            print("saving image for best error: " + str(lossAvg.item()))
            bestErrorScore = lossAvg
            info = PngImagePlugin.PngInfo()
            info.add_text('comment', f'{cmdLineArgs.args.prompts}')
            TF.to_pil_image(out[0].cpu()).save( "lowest_error_" + cmdLineArgs.args.output, pnginfo=info)

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
            z.copy_(z.maximum(z_min).minimum(z_max))


loss_idx = []
i = 0 # Iteration counter
j = 0 # Zoom video frame counter
p = 1 # Phrase counter
smoother = 0 # Smoother counter
this_video_frame = 0 # for video styling

# Messing with learning rate / optimisers
#variable_lr = args.step_size
#optimiser_list = [['Adam',0.075],['AdamW',0.125],['Adagrad',0.2],['Adamax',0.125],['DiffGrad',0.075],['RAdam',0.125],['RMSprop',0.02]]

# Creates a GradScaler once at the beginning of training.
scaler = GradScaler()

# Do it
try:
    with tqdm() as pbar:
        while True:            
            # Change generated image
            if cmdLineArgs.args.make_zoom_video:
                if i % cmdLineArgs.args.zoom_frequency == 0:
                    out = synth(z)
                    
                    # Save image
                    img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:,:,:]
                    img = np.transpose(img, (1, 2, 0))
                    imageio.imwrite('./steps/' + str(j) + '.png', np.array(img))

                    # Time to start zooming?                    
                    if cmdLineArgs.args.zoom_start <= i:
                        # Convert z back into a Pil image                    
                        #pil_image = TF.to_pil_image(out[0].cpu())
                        
                        # Convert NP to Pil image
                        pil_image = Image.fromarray(np.array(img).astype('uint8'), 'RGB')
                                                
                        # Zoom
                        if cmdLineArgs.args.zoom_scale != 1:
                            pil_image_zoom = zoom_at(pil_image, sideX/2, sideY/2, cmdLineArgs.args.zoom_scale)
                        else:
                            pil_image_zoom = pil_image
                        
                        # Shift - https://pillow.readthedocs.io/en/latest/reference/ImageChops.html
                        if cmdLineArgs.args.zoom_shift_x or cmdLineArgs.args.zoom_shift_y:
                            # This one wraps the image
                            pil_image_zoom = ImageChops.offset(pil_image_zoom, cmdLineArgs.args.zoom_shift_x, cmdLineArgs.args.zoom_shift_y)
                        
                        # Convert image back to a tensor again
                        pil_tensor = TF.to_tensor(pil_image_zoom)
                        
                        # Re-encode
                        z, *_ = vqganModel.encode(pil_tensor.to(vqganDevice).unsqueeze(0) * 2 - 1)
                        z_orig = z.clone()
                        z.requires_grad_(True)

                        # Re-create optimiser
                        opt = get_opt(cmdLineArgs.args.optimiser, cmdLineArgs.args.step_size)
                    
                    # Next
                    j += 1
            
            # Change text prompt
            if cmdLineArgs.args.prompt_frequency > 0:
                if i % cmdLineArgs.args.prompt_frequency == 0 and i > 0:
                    # In case there aren't enough phrases, just loop
                    if p >= len(all_phrases):
                        p = 0
                    
                    pMs = []
                    cmdLineArgs.args.prompts = all_phrases[p]

                    # Show user we're changing prompt                                
                    print(cmdLineArgs.args.prompts)
                    
                    for prompt in cmdLineArgs.args.prompts:
                        txt, weight, stop = split_prompt(prompt)
                        embed = clipPerceptor.encode_text(clip.tokenize(txt).to(clipDevice)).float()
                        pMs.append(Prompt(embed, weight, stop).to(clipDevice))

                                        
                    '''
                    # Smooth test
                    smoother = args.zoom_frequency * 15 # smoothing over x frames
                    variable_lr = args.step_size * 0.25
                    opt = get_opt(args.optimiser, variable_lr)
                    '''
                    
                    p += 1
            
            '''
            if smoother > 0:
                if smoother == 1:
                    opt = get_opt(args.optimiser, args.step_size)
                smoother -= 1
            '''
            
            '''
            # Messing with learning rate / optimisers
            if i % 225 == 0 and i > 0:
                variable_optimiser_item = random.choice(optimiser_list)
                variable_optimiser = variable_optimiser_item[0]
                variable_lr = variable_optimiser_item[1]
                
                opt = get_opt(variable_optimiser, variable_lr)
                print("New opt: %s, lr= %f" %(variable_optimiser,variable_lr)) 
            '''
            

            # Training time
            train(i)

            if i == cmdLineArgs.args.max_iterations:
                out = synth(z)
                    
                # Save image
                img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:,:,:]
                img = np.transpose(img, (1, 2, 0))
                imageio.imwrite(cmdLineArgs.args.output, np.array(img))                
            
            
            # Ready to stop yet?
            if i == cmdLineArgs.args.max_iterations:
                if cmdLineArgs.args.log_clip:    
                    # write one for the console
                    WriteLogClipResults(out)
                	#write once to a file for easy grabbing outside of this script                
                    text_file = open(cmdLineArgs.args.output + ".txt", "w")
                    sys.stdout = text_file
                    WriteLogClipResults(out)
                    sys.stdout = sys.stdout 
                    text_file.close()

                if not cmdLineArgs.args.video_style_dir:
                    # we're done
                    break
                else:                    
                    if this_video_frame == (num_video_frames - 1):
                        # we're done
                        make_styled_video = True
                        break
                    else:
                        # Next video frame
                        this_video_frame += 1

                        # Reset the iteration count
                        i = -1
                        pbar.reset()
                                                
                        # Load the next frame, reset a few options - same filename, different directory
                        cmdLineArgs.args.init_image = video_frame_list[this_video_frame]
                        print("Next frame: ", cmdLineArgs.args.init_image)

                        if cmdLineArgs.args.seed is None:
                            seed = torch.seed()
                        else:
                            seed = cmdLineArgs.args.seed  
                        torch.manual_seed(seed)
                        print("Seed: ", seed)

                        filename = os.path.basename(cmdLineArgs.args.init_image)
                        cmdLineArgs.args.output = os.path.join(cwd, "steps", filename)

                        # Load and resize image
                        img = Image.open(cmdLineArgs.args.init_image)
                        pil_image = img.convert('RGB')
                        pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
                        pil_tensor = TF.to_tensor(pil_image)
                        
                        # Re-encode
                        z, *_ = model.encode(pil_tensor.to(device).unsqueeze(0) * 2 - 1)
                        z_orig = z.clone()
                        z.requires_grad_(True)

                        # Re-create optimiser
                        opt = get_opt(cmdLineArgs.args.optimiser, cmdLineArgs.args.step_size)

            i += 1
            pbar.update()
except KeyboardInterrupt:
    pass

# All done :)

# Video generation
if cmdLineArgs.args.make_video or cmdLineArgs.args.make_zoom_video:
    init_frame = 1      # Initial video frame
    if cmdLineArgs.args.make_zoom_video:
        last_frame = j
    else:
        last_frame = i  # This will raise an error if that number of frames does not exist.

    length = cmdLineArgs.args.video_length # Desired time of the video in seconds

    min_fps = 10
    max_fps = 60

    total_frames = last_frame-init_frame

    frames = []
    tqdm.write('Generating video...')
    for i in range(init_frame,last_frame):
        temp = Image.open("./steps/"+ str(i) +'.png')
        keep = temp.copy()
        frames.append(keep)
        temp.close()
    
    if cmdLineArgs.args.output_video_fps > 9:
        # Hardware encoding and video frame interpolation
        print("Creating interpolated frames...")
        ffmpeg_filter = f"minterpolate='mi_mode=mci:me=hexbs:me_mode=bidir:mc_mode=aobmc:vsbmc=1:mb_size=8:search_param=32:fps={cmdLineArgs.args.output_video_fps}'"
        output_file = re.compile('\.png$').sub('.mp4', cmdLineArgs.args.output)
        try:
            p = Popen(['ffmpeg',
                       '-y',
                       '-f', 'image2pipe',
                       '-vcodec', 'png',
                       '-r', str(cmdLineArgs.args.input_video_fps),               
                       '-i',
                       '-',
                       '-b:v', '10M',
                       '-vcodec', 'h264_nvenc',
                       '-pix_fmt', 'yuv420p',
                       '-strict', '-2',
                       '-filter:v', f'{ffmpeg_filter}',
                       '-metadata', f'comment={cmdLineArgs.args.prompts}',
                   output_file], stdin=PIPE)
        except FileNotFoundError:
            print("ffmpeg command failed - check your installation")
        for im in tqdm(frames):
            im.save(p.stdin, 'PNG')
        p.stdin.close()
        p.wait()
    else:
        # CPU
        fps = np.clip(total_frames/length,min_fps,max_fps)
        output_file = re.compile('\.png$').sub('.mp4', cmdLineArgs.args.output)
        try:
            p = Popen(['ffmpeg',
                       '-y',
                       '-f', 'image2pipe',
                       '-vcodec', 'png',
                       '-r', str(fps),
                       '-i',
                       '-',
                       '-vcodec', 'libx264',
                       '-r', str(fps),
                       '-pix_fmt', 'yuv420p',
                       '-crf', '17',
                       '-preset', 'veryslow',
                       '-metadata', f'comment={cmdLineArgs.args.prompts}',
                       output_file], stdin=PIPE)
        except FileNotFoundError:
            print("ffmpeg command failed - check your installation")        
        for im in tqdm(frames):
            im.save(p.stdin, 'PNG')
        p.stdin.close()
        p.wait()     
