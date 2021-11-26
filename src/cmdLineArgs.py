import argparse
import torch
from torch.cuda import get_device_properties
import json
import copy

# this is used globally until i clean this all up
args = None

def init():
    global args


    # Check for GPU and reduce the default image size if low VRAM
    default_image_size = 512  # >8GB VRAM
    if not torch.cuda.is_available():
        default_image_size = 256  # no GPU found
    elif get_device_properties(0).total_memory <= 2 ** 33:  # 2 ** 33 = 8,589,934,592 bytes = 8 GB
        default_image_size = 318  # <8GB VRAM

    # Create the parser
    vq_parser = argparse.ArgumentParser(description='Image generation using VQGAN+CLIP')

    # Add the arguments
    vq_parser.add_argument("-p",    "--prompts", type=str, help="Text prompts", default=None, dest='prompts')
    vq_parser.add_argument("-ip",   "--image_prompts", type=str, help="Image prompts / target image", default=[], dest='image_prompts')
    vq_parser.add_argument("-i",    "--iterations", type=int, help="Number of iterations", default=500, dest='max_iterations')    
    vq_parser.add_argument("-stats","--stats_every", type=int, help="stats display frequency", default=50, dest='display_freq')
    vq_parser.add_argument("-s",    "--size", nargs=2, type=int, help="Image size (width height) (default: %(default)s)", default=[default_image_size,default_image_size], dest='size')
    vq_parser.add_argument("-ii",   "--init_image", type=str, help="Initial image", default=None, dest='init_image')
    vq_parser.add_argument("-in",   "--init_noise", type=str, help="Initial noise image (pixels or gradient)", default=None, dest='init_noise')
    vq_parser.add_argument("-iw",   "--init_weight", type=float, help="Initial weight", default=0., dest='init_weight')
    vq_parser.add_argument("-m",    "--clip_model", type=str, help="CLIP model (e.g. ViT-B/32, ViT-B/16)", default='ViT-B/32', dest='clip_model')
    vq_parser.add_argument("-conf", "--vqgan_config", type=str, help="VQGAN config", default=f'checkpoints/vqgan_imagenet_f16_16384.yaml', dest='vqgan_config')
    vq_parser.add_argument("-ckpt", "--vqgan_checkpoint", type=str, help="VQGAN checkpoint", default=f'checkpoints/vqgan_imagenet_f16_16384.ckpt', dest='vqgan_checkpoint')
    vq_parser.add_argument("-nps",  "--noise_prompt_seeds", nargs="*", type=int, help="Noise prompt seeds", default=[], dest='noise_prompt_seeds')
    vq_parser.add_argument("-npw",  "--noise_prompt_weights", nargs="*", type=float, help="Noise prompt weights", default=[], dest='noise_prompt_weights')
    vq_parser.add_argument("-lr",   "--learning_rate", type=float, help="Learning rate", default=0.1, dest='step_size')
    vq_parser.add_argument("-sd",   "--seed", type=int, help="Seed", default=None, dest='seed')
    vq_parser.add_argument("-opt",  "--optimiser", type=str, help="Optimiser", choices=['Adam','AdamW','Adagrad','Adamax','DiffGrad','AdamP','RMSprop','MADGRAD'], default='Adam', dest='optimiser')    
    vq_parser.add_argument("-cpe",  "--change_prompt_every", type=int, help="Prompt change frequency", default=0, dest='prompt_frequency')    
    vq_parser.add_argument("-aug",  "--augments", nargs='+', action='append', type=str, choices=['None','Ji','Sh','Gn','Pe','Ro','Af','Et','Ts','Cr','Er','Re'], help="Enabled augments (latest vut method only)", default=[], dest='augments')
    vq_parser.add_argument("-cd",   "--cuda_device", type=str, help="Cuda device to use", default="cuda:0", dest='cuda_device')


    vq_parser.add_argument("-d",    "--deterministic", type=int, default=1, help="Determinism: 0 ( none ), 1 ( some, default ), 2 ( as much as possible )", dest='deterministic')

    # cuts and pooling
    vq_parser.add_argument("-cutm", "--cut_method", type=str, help="Cut method", choices=['original','nerdyNoPool','nerdy','squish','latest','test','growFromCenter'], default='latest', dest='cut_method')
    vq_parser.add_argument("-cuts", "--num_cuts", type=int, help="Number of cuts", default=32, dest='cutn')
    vq_parser.add_argument("-cutp", "--cut_power", type=float, help="Cut power", default=1., dest='cut_pow')

    vq_parser.add_argument("-cutsize",    "--cut_size", nargs=2, type=int, help="Cut size (width height) (clip controlled)", default=[0,0], dest='cut_size')

    # manage output files and other logged data 
    vq_parser.add_argument("-od",   "--output_dir", type=str, help="Output filename", default="./output/", dest='output_dir')
    vq_parser.add_argument("-o",    "--output", type=str, help="Output filename", default="output.png", dest='output')    
    vq_parser.add_argument("-se",   "--save_every", type=int, help="Save image iterations", default=50, dest='save_freq')
    vq_parser.add_argument("-sb",   "--save_best", help="Save the best scored image", action='store_true', dest='save_best')

    # attempt to use mixed precision mode here
    # need to hunt down causes of the decoder produces inf's/nan's. current hack is to replace them with min/max floats, slower and produces poorer results than 32 bit.
    vq_parser.add_argument("-usemix", "--use_mixed",  action='store_true', help="mixed precision reduces memory size greatly, Augmentations do not work in this mode, yet", dest='use_mixed_precision')

    # this works, can go from 400x400 to 575x400 on an 8GB card being used as a display device
    # time to crunch went up from 1.07 per iter to 7.5s per iteration, cant be used in conjunction with mixed precision mode ( yet )
    vq_parser.add_argument("-clipcpu",   "--clip_cpu",  action='store_true', help="forces the clip model into the cpu. slows things down but frees up memory so you can make larger images on low VRAM cards", 
                            dest='clip_cpu')

    # helpful for tracking down various floating point errors 
    vq_parser.add_argument("-ac",   "--anomalyChecker",  action='store_true', help="enabled the pyTorch anomaly checker for nan's, useful in mixed precision mode which has had some issues", 
                            dest='anomaly_checker')

    # simple memory logger
    vq_parser.add_argument("-lm",   "--logMem",  action='store_true', help="log memory usage every checkin", dest='log_mem')

    # writes one shot clip and a sort of prompt stat check using clip
    vq_parser.add_argument("-lcp",   "--logClipProbabilities",  action='store_true', dest='log_clip')
    vq_parser.add_argument("-lcos",  "--logClipOneShotGuesses",  action='store_true', dest='log_clip_oneshot')

    #allow configs from json files / save to json file for later preservation
    vq_parser.add_argument('--save_json', help='Save settings to file in json format. Ignored in json file')
    vq_parser.add_argument('--save_json_strip_defaults', action='store_true', help='remove default / unset settings when saving config')
    vq_parser.add_argument('--save_json_strip_misc', action='store_true', help='remove misc settings when saving config to use as a command')
    vq_parser.add_argument('--load_json', help='Load settings from file in json format. Command line options override values in file.')
    vq_parser.add_argument('--convert_to_json_cmd', action='store_true', help='only load/save back to json for use as a cmd')




    # Execute the parse_args() method
    args = vq_parser.parse_args()

    if args.load_json:
        with open(args.load_json, 'rt') as f:
            t_args = argparse.Namespace()
            jsonObj = json.load(f)
            
            # strip out the save/load json params
            jsonObj.pop("save_json", None)
            jsonObj.pop("load_json", None)
            jsonObj.pop("save_json_strip_defaults", None)

            t_args.__dict__.update(jsonObj)
            args = vq_parser.parse_args(namespace=t_args)            

    # Optional: support for saving settings into a json file
    if args.save_json:
        inputArgs = vars(copy.deepcopy(args))

        # strip out the save/load json params
        inputArgs.pop("save_json", None)
        inputArgs.pop("load_json", None)
        inputArgs.pop("save_json_strip_defaults", None)       
        inputArgs.pop("save_json_strip_misc", None)
        inputArgs.pop("convert_to_json_cmd", None)


        if args.save_json_strip_defaults:
            # lets try to strip out all default options that are the same (since versionc hanges may change defaults)            
            defaultArgs = vars(vq_parser.parse_args(''))
        
            # generate output dictionary
            inputArgs = {key:val for key, val in inputArgs.items() if not key in defaultArgs or defaultArgs[key] != val}


        if args.save_json_strip_misc:
            #pull out user specific stuff here that will almost always be controlled via commandline, so we can make a clean command file
            inputArgs.pop("prompts", None)
            inputArgs.pop("seed", None)
            inputArgs.pop("image_prompts", None)
            inputArgs.pop("stats_every", None)
            inputArgs.pop("output_dir", None)
            inputArgs.pop("output", None)
            inputArgs.pop("save_every", None)
            inputArgs.pop("save_best", None)
            inputArgs.pop("anomalyChecker", None)
            inputArgs.pop("logMem", None)
            inputArgs.pop("logClipProbabilities", None)            

        with open(args.save_json, 'wt') as f:
            json.dump(inputArgs, f, indent=4)
