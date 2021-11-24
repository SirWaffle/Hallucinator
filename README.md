# original VQGAN-CLIP implementation ( forked from nerdyrodent )

many thanks to  <https://github.com/nerdyrodent/VQGAN-CLIP> ! More information and sample images can be seen on the original github



# Hallucinator
I have modified the original readme to reflect my repo - the first readme was well done and obtained from the original fork.

## Warnings
-  I am an experienced programmer, but new to python, so my code may be very non-pythonic and weird in many places
-  I am still learning the basic of ML, so expect hacky test things and weirdness
-  I am treating this as my own private repo without concerns of anyone that might look at this code, so things may change without warning and with low quality 

## Change list from original repository
- minor memory usage reductions
- minor performance improvements
- can now fully generate deterministic images, although its slower and can't be used with pooling and other features
- code refactoring
- various new commandline options
- ability to write out / load in json configs for common sets of command line options, which can be overriden via commandline. nice to use for creating various sets of parameters for different genreation techniques
- addition of mixed precision mode to save more memory ( but the output isnt very good yet )
- more options for where / how often output gets saved and written
- removal of video generation from scripts to reduce clutter ( i use external tools for this )
- stats, memory usage logging
- clip analysis logging which may or may not be correct
- cut method modifications for higher res images

## roadmap
- masking to prevent modifications in certain places, masking to provide specific prompts in certain locations
- integration via plugins to art programs for itneractive generation
- interactive server mode for a dedicated interactive instance on a local machine



# ==== Instructions ======

A repo for running VQGAN+CLIP locally. This started out as a Katherine Crowson VQGAN+CLIP derived Google colab notebook.

Original notebook: [![Open In Colab][colab-badge]][colab-notebook]

[colab-notebook]: <https://colab.research.google.com/drive/1ZAus_gn2RhTZWzOWUpPERNC0Q8OhZRTZ>
[colab-badge]: <https://colab.research.google.com/assets/colab-badge.svg>


Environment:

* Tested on Windows 10
* PyTorch 1.10.0
* GPUs: Nvidia RTX A6000, 1080 TI, and 1070 Mobile
* Typical VRAM requirements:
  * 24 GB for a 900x900 image
  * 10 GB for a 512x512 image
  * 8 GB for a 380x380 image

## Set up

This example uses [Anaconda](https://www.anaconda.com/products/individual#Downloads) to manage virtual Python environments.

Create a new virtual Python environment for VQGAN-CLIP:

```sh
conda create --name hallucinator python=3.9
conda activate hallucinator
```

Install Pytorch in the new enviroment:
<https://pytorch.org/get-started/locally/>
will generate a command line for you to install the latest pytorch. I use the conda install, but the pip command version should also work


Install other required Python packages:

```sh
pip install ftfy regex tqdm omegaconf pytorch-lightning IPython kornia imageio imageio-ffmpeg einops torch_optimizer transformers
```

Clone required repositories:

```sh
git clone "https://github.com/SirWaffle/Hallucinator"
git clone "https://github.com/SirWaffle/CLIP"
cd VQGAN-CLIP
git clone "https://github.com/SirWaffle/taming-transformers.git"
```

Notes: 
- It is not neccessary to use my forks of CLIP or taming-transformers, i only have those to control what changes filter to my project
- If you want to mess with mixed precision mode for memory savings, you will need my fork of taming-transformers
- In my development environment taming-transformers is present in the local directory, and so aren't present in the `requirements.txt` or `vqgan.yml` files.
- In my environment, CLIP sits alongside my project directory, but i use it from source, not install

As an alternative, you can also pip install taming-transformers and CLIP.

You will also need at least 1 VQGAN pretrained model. E.g.

```sh
mkdir checkpoints

curl -L -o checkpoints/vqgan_imagenet_f16_16384.yaml -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1' #ImageNet 16384
curl -L -o checkpoints/vqgan_imagenet_f16_16384.ckpt -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1' #ImageNet 16384
```

The `download_models.sh` script is an optional way to download a number of models. By default, it will download just 1 model.

See <https://github.com/CompVis/taming-transformers#overview-of-pretrained-models> for more information about VQGAN pre-trained models, including download links.

By default, the model .yaml and .ckpt files are expected in the `checkpoints` directory.
See <https://github.com/CompVis/taming-transformers> for more information on datasets and models.

### Using the CPU

If no graphics card can be found, the CPU is automatically used and a warning displayed.

Regardless of an available graphics card, the CPU can also be used by adding this command line argument: `-cd cpu`

This works with the CUDA version of Pytorch, even without CUDA drivers installed, but doesn't seem to work with ROCm as of now.

### Uninstalling

Remove the Python enviroment:

```sh
conda remove --name hallucinator --all
```

and delete the `hallucinator` directory.

## Run

To generate images from text, specify your text prompt as shown in the example below:

```sh
python generate.py --prompt "A painting of an apple in a fruit bowl"
```

## Multiple prompts

Text and image prompts can be split using the pipe symbol in order to allow multiple prompts.
You can also use a colon followed by a number to set a weight for that prompt. For example:

```sh
python generate.py --prompt "A painting of an apple in a fruit bowl | psychedelic | surreal:0.5 | weird:0.25"
```

Image prompts can be split in the same way. For example:

```sh
python generate.py --prompt "A picture of a bedroom with a portrait of Van Gogh" -ip "samples/VanGogh.jpg | samples/Bedroom.png"
```

### Story mode

Sets of text prompts can be created using the caret symbol, in order to generate a sort of story mode. For example:

```sh
python generate.py --prompt "A painting of a sunflower|photo:-1 ^ a painting of a rose ^ a painting of a tulip ^ a painting of a daisy flower ^ a photograph of daffodil"
```


## "Style Transfer"

An input image with style text and a low number of iterations can be used create a sort of "style transfer" effect. For example:

```sh
python generate.py --prompt "A painting in the style of Picasso" -ii samples/VanGogh.jpg -i 80 -se 10 -opt AdamW -lr 0.25
```


## Advanced options

To view the available options, use "-h"

```sh
python generate.py -h
```

you can also view all command line options in the ./src/cmdLineArgs.py file



## Citations

```bibtex
@misc{unpublished2021clip,
    title  = {CLIP: Connecting Text and Images},
    author = {Alec Radford, Ilya Sutskever, Jong Wook Kim, Gretchen Krueger, Sandhini Agarwal},
    year   = {2021}
}
```

```bibtex
@misc{esser2020taming,
      title={Taming Transformers for High-Resolution Image Synthesis}, 
      author={Patrick Esser and Robin Rombach and Björn Ommer},
      year={2020},
      eprint={2012.09841},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

Katherine Crowson - <https://github.com/crowsonkb>

Nerdy Rodent - <https://github.com/nerdyrodent/VQGAN-CLIP>
