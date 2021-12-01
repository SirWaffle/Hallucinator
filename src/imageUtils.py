import numpy as np
from PIL import Image
import math

import torch
from torch import nn
from torch.nn import functional as F

def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)


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



# Various functions and classes
def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]

# Used in older MakeCutouts
# resample is non-deterministic due to interpolate bicubic
# F.pad is non determinsitic...
def resample(input: torch.Tensor, sizeYX, align_corners=True, deterministic = False):
    n, c, h, w = input.shape
    dh, dw = sizeYX

    input = input.view([n * c, 1, h, w])

    if not deterministic:
        if dh < h:
            kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
            pad_h = (kernel_h.shape[0] - 1) // 2
            input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
            input = F.conv2d(input, kernel_h[None, None, :, None])

        if dw < w:
            kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
            pad_w = (kernel_w.shape[0] - 1) // 2
            input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
            input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.view([n, c, h, w])

    if deterministic:
        #docs claim nearest and area are deterministic
        return F.interpolate(input, sizeYX, mode='nearest') #align corners has to be false for linear due to some issues
    else:
        return F.interpolate(input, sizeYX, mode='bicubic', align_corners=align_corners)    
        