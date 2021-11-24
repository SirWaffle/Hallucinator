# VQGAN-CLIP ( forked from nerdyrodent )

This was forked from <https://github.com/nerdyrodent/VQGAN-CLIP>, so that I could tinker with it to my own ends. 


# Hallucinator VQGAN+CLIP

I have modified the original readme to reflect my repo - the first readme was well done and obtained from the original fork.

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
pip install ftfy regex tqdm omegaconf pytorch-lightning IPython kornia imageio imageio-ffmpeg einops torch_optimizer
```

Clone required repositories:

```sh
git clone "https://github.com/SirWaffle/Hallucinator"
git clone "https://github.com/SirWaffle/CLIP"
cd VQGAN-CLIP
git clone "https://github.com/SirWaffle/taming-transformers.git"
```

Note: In my development environment both CLIP and taming-transformers are present in the local directory, and so aren't present in the `requirements.txt` or `vqgan.yml` files.

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
conda remove --name vqgan --all
```

and delete the `VQGAN-CLIP` directory.

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

To view the available options, use "-h".

```sh
python generate.py -h
```

you can also view all command line options in the ./src/cmdLineArgs.py file

## Troubleshooting

### CUSOLVER_STATUS_INTERNAL_ERROR

For example:

`RuntimeError: cusolver error: CUSOLVER_STATUS_INTERNAL_ERROR, when calling cusolverDnCreate(handle)`

Make sure you have specified the correct size for the image. For more information please refer to [#6](/issues/6)

### RuntimeError: CUDA out of memory

For example:

`RuntimeError: CUDA out of memory. Tried to allocate 150.00 MiB (GPU 0; 23.70 GiB total capacity; 21.31 GiB already allocated; 78.56 MiB free; 21.70 GiB reserved in total by PyTorch)`

Your request doesn't fit into your GPU's VRAM. Reduce the image size and/or number of cuts.


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
