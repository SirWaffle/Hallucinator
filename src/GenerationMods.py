# classes for adding modifications to images during training
# such as the zoom in stuff
# or masking parts of original image onto currently generated image

import abc
from enum import Enum, auto
from src import ImageUtils
import numpy as np

import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from PIL import ImageFile, Image, ImageChops
ImageFile.LOAD_TRUNCATED_IMAGES = True

 
# TODO: this may be better off not being a sign of 'when' the mod should apply, since most will be during the training loop
# but what the mod applies to, and therefor what needs updating when the mod does its thing
# ex: 
#   ImageMod requires re-encoding and redoing the zer, 
#   prompt mods need to re-encode with clip, 
#
# and in the future, these mods will be turned into more of a 'command' pattern, to modify stuff from art program plugins
# so there will be things like:
#    LoadNewImage, ResetOptimizer, ChangeLearningRate, ChangeOptimizer, ChangeCutouts, all of which require different things to be redone / different ways to handle
class GenerationModStage(Enum):
    PreTrain = auto() #pretraining step, mods happen before the next training step
    FinishedGeneration = auto() #happens when the generation of the image is finished


class IGenerationMod(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'OnPreTrain') and 
                callable(subclass.OnPreTrain) or 
                NotImplemented)

    def __init__(self, GenJob, ModStage:GenerationModStage = GenerationModStage.PreTrain):
        super().__init__()

        self.GenJob = GenJob  
        self.ModStage = ModStage      

    @abc.abstractmethod
    def Initialize(self):
        raise NotImplementedError

    def ShouldApply(self, stage: GenerationModStage, iteration: int ) -> bool:
        return stage.value == self.ModStage.value

    @abc.abstractmethod
    def OnPreTrain(self, iteration: int ):
        raise NotImplementedError



class GenerationModContainer:
    def __init__(self, mod:IGenerationMod, startIt: int = 0, endIt: int = 9999999999, freq: int = 10):
        self.freq = freq
        self.startIt = startIt
        self.endIt = endIt
        self.mod = mod

    def ShouldApply(self, stage: GenerationModStage, iteration: int ) -> bool:
        if self.mod.ShouldApply(stage, iteration):
            if iteration >= self.startIt and iteration <= self.endIt:
                iterationDelta = iteration - self.startIt
                if  iterationDelta % self.freq == 0:
                    return True
        return False

    def OnPreTrain(self, iteration: int ):
        self.mod.OnPreTrain(iteration)



# adds a mask taht can be repeatedly pasted on the image
# from the initial source image
class OriginalImageMask(IGenerationMod):

    def __init__(self, GenJob, maskPath: str = ''):
        super().__init__(GenJob)

        self.maskPath:str = maskPath
        self.original_image_tensor:torch.Tensor = None      
        self.image_mask_tensor:torch.Tensor = None          
        self.image_mask_tensor_invert:torch.Tensor = None 

    def Initialize(self):
        with torch.inference_mode():
            original_pil = self.GenJob.GerCurrentImageAsPIL()
            self.original_image_tensor = TF.to_tensor(original_pil).to(self.GenJob.vqganDevice)

            img = Image.open(self.maskPath)
            pil_image = img.convert('RGB')
        
            image_mask_np  = np.asarray(pil_image)

            #makes float32 mask
            self.image_mask_tensor = TF.to_tensor(image_mask_np).to(self.GenJob.vqganDevice)

            #make boolean masks
            self.image_mask_tensor_invert = torch.logical_not( self.image_mask_tensor )
            self.image_mask_tensor = torch.logical_not( self.image_mask_tensor_invert )


    def OnPreTrain(self, iteration: int ):
        with torch.inference_mode():
            curQuantImg = self.GenJob.GetCurrentImageSynthed()

            #this removes the first dim sized 1 to match the rest
            curQuantImg = torch.squeeze(curQuantImg)

            keepCurrentImg = curQuantImg * self.image_mask_tensor_invert.int().float()
            keepOrig = self.original_image_tensor * self.image_mask_tensor.int().float()
            pil_tensor = keepCurrentImg + keepOrig

        # Re-encode original?
        self.GenJob.quantizedImage, *_ = self.GenJob.vqganModel.encode(pil_tensor.to(self.GenJob.vqganDevice).unsqueeze(0) * 2 - 1)
        #self.GenJob.original_quantizedImage = self.GenJob.quantizedImage.detach()
        
        self.GenJob.quantizedImage.requires_grad_(True)
        self.GenJob.optimizer = self.GenJob.hallucinatorInst.get_optimizer(self.GenJob.quantizedImage, self.GenJob.optimizerName, self.GenJob.step_size)





# from original inmplementation of image zoom from nerdyRodent
class ImageZoomer(IGenerationMod):
    #fucking python has no maxint to use as a large value, annoying
    def __init__(self, GenJob,zoom_scale: float = 0.99, zoom_shift_x: int = 0, zoom_shift_y: int = 0):
        super().__init__(GenJob)

        ##need to make these configurable
        self.zoom_scale = zoom_scale
        self.zoom_shift_x = zoom_shift_x
        self.zoom_shift_y = zoom_shift_y

    def Initialize(self):
        pass


    def OnPreTrain(self, iteration: int ):
        with torch.inference_mode():
            out = self.GenJob.GetCurrentImageSynthed()
            
            # Save image
            img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:,:,:]
            img = np.transpose(img, (1, 2, 0))
            
            # Convert NP to Pil image
            pil_image = Image.fromarray(np.array(img).astype('uint8'), 'RGB')
                                    
            # Zoom
            if self.zoom_scale != 1:
                pil_image_zoom = ImageUtils.zoom_at(pil_image, self.GenJob.ImageSizeX/2, self.GenJob.ImageSizeY/2, self.zoom_scale)
            else:
                pil_image_zoom = pil_image
            
            # Shift - https://pillow.readthedocs.io/en/latest/reference/ImageChops.html
            if self.zoom_shift_x or self.zoom_shift_y:
                # This one wraps the image
                pil_image_zoom = ImageChops.offset(pil_image_zoom, self.zoom_shift_x, self.zoom_shift_y)
            
            # Convert image back to a tensor again
            pil_tensor = TF.to_tensor(pil_image_zoom)

        # Re-encode original?
        self.GenJob.quantizedImage, *_ = self.GenJob.vqganModel.encode(pil_tensor.to(self.GenJob.vqganDevice).unsqueeze(0) * 2 - 1)
        #self.GenJob.original_quantizedImage = self.GenJob.quantizedImage.detach()
        
        self.GenJob.quantizedImage.requires_grad_(True)
        self.GenJob.optimizer = self.GenJob.hallucinatorInst.get_optimizer(self.GenJob.quantizedImage, self.GenJob.optimizerName, self.GenJob.step_size)





# faster tensor based image zoomer, but only zooms in for now
class ImageZoomInFast(IGenerationMod):
    #fucking python has no maxint to use as a large value, annoying
    def __init__(self, GenJob, zoom_scale: float = 1.02, normalizedZoomPointX: float = 0.5, normalizedZoomPointY: float = 0.5):
        super().__init__(GenJob)

        ##need to make these configurable
        self.zoom_scale = zoom_scale
        self.normalizedZoomPointX = normalizedZoomPointX
        self.normalizedZoomPointY = normalizedZoomPointY

        if self.zoom_scale < 1.0:
            print("Error: zoom_scale in ImageZoomInFast mod too low")

    def Initialize(self):
        pass

    def OnPreTrain(self, iteration: int ):
        with torch.inference_mode():
            imgTensor = self.GenJob.GetCurrentImageSynthed()

            n, c, h, w = imgTensor.shape

            size_x = int( ( 1.0 / self.zoom_scale ) * w )
            size_y = int( ( 1.0 / self.zoom_scale ) * h )
            offsetx = int( ( w - size_x ) / 2 )
            offsety = int( ( h - size_y ) / 2 )

            offsetx = int( offsetx * ( 2 * self.normalizedZoomPointX ) )
            offsety = int( offsety * ( 2 * self.normalizedZoomPointY ) )

            zoomPortion = imgTensor[:, :, offsety:offsety + size_y, offsetx:offsetx + size_x]

            # TODO: this is currently non-deterministic
            zoomPortion = F.interpolate(zoomPortion, (h, w), mode='bicubic', align_corners=True)  
            #zoomPortion = ImageUtils.resample(zoomPortion, (h, w))

            # TODO: can probably remove this and the unsqueeze below...
            imgTensor = torch.squeeze(zoomPortion)

        # Re-encode original?
        self.GenJob.quantizedImage, *_ = self.GenJob.vqganModel.encode(imgTensor.to(self.GenJob.vqganDevice).unsqueeze(0) * 2 - 1)
        #self.GenJob.original_quantizedImage = self.GenJob.quantizedImage.detach()

        self.GenJob.quantizedImage.requires_grad_(True)
        self.GenJob.optimizer = self.GenJob.hallucinatorInst.get_optimizer(self.GenJob.quantizedImage, self.GenJob.optimizerName, self.GenJob.step_size)





class ImageRotate(IGenerationMod):
    def __init__(self, GenJob, angle: int = 1):
        super().__init__(GenJob)

        self.angle = angle

    def Initialize(self):
        pass


    def OnPreTrain(self, iteration: int ):
        with torch.inference_mode():
            curQuantImg = self.GenJob.GetCurrentImageSynthed()

            #this removes the first dim sized 1 to match the rest
            curQuantImg = torch.squeeze(curQuantImg)

            pil_tensor = TF.rotate(curQuantImg, self.angle)

        # Re-encode original?
        self.GenJob.quantizedImage, *_ = self.GenJob.vqganModel.encode(pil_tensor.to(self.GenJob.vqganDevice).unsqueeze(0) * 2 - 1)
        #self.GenJob.original_quantizedImage = self.GenJob.quantizedImage.detach()
        
        self.GenJob.quantizedImage.requires_grad_(True)
        self.GenJob.optimizer = self.GenJob.hallucinatorInst.get_optimizer(self.GenJob.quantizedImage, self.GenJob.optimizerName, self.GenJob.step_size)                



class ChangePromptMod(IGenerationMod):
    def __init__(self, GenJob, prompt:str, clearOtherPrompts:bool = True):
        super().__init__(GenJob)

        self.prompt = prompt
        self.clearOtherPrompts = clearOtherPrompts

    def Initialize(self):
        pass


    def OnPreTrain(self, iteration: int ):
        if self.clearOtherPrompts == True:
            self.GenJob.embededPrompts = []

        print('Changing prompt to: "' + self.prompt + '", from ' + str(self))
        self.GenJob.EmbedTextPrompt(self.prompt)    



class AddPromptMask(IGenerationMod):
    def __init__(self, GenJob, prompt:str, maskImageFileName:str, dilateMaskAmount:int = 10, clearOtherPrompts:bool = False, cacheImageOnInit:bool = True):
        super().__init__(GenJob)

        self.prompt = prompt
        self.maskImageFileName = maskImageFileName
        self.clearOtherPrompts = clearOtherPrompts
        self.cacheImageOnInit = cacheImageOnInit
        self.imageTensor:torch.Tensor = None
        self.dilateMaskAmount = dilateMaskAmount

    def Initialize(self):
        with torch.inference_mode():
            # load image into tensor? or we can do it when its time to add the mod, memory now vs. performance later...
            if self.cacheImageOnInit:
                #cache the image as a tensor here
                self.imageTensor = ImageUtils.loadImageToTensor(self.maskImageFileName, self.GenJob.ImageSizeX, self.GenJob.ImageSizeY )

                #dialate
                if self.dilateMaskAmount:
                    struct_ele = torch.FloatTensor(1,1,self.dilateMaskAmount,self.dilateMaskAmount).fill_(1).to(self.GenJob.vqganDevice)
                    self.imageTensor = F.conv2d(self.imageTensor,struct_ele,padding='same')

                #resize masks to output size
                self.imageTensor = F.interpolate(self.imageTensor,(self.GenJob.toksY * 16, self.GenJob.toksX * 16))

                #make binary
                self.imageTensor[self.imageTensor>0.1]=1

    def OnPreTrain(self, iteration: int ):
        if self.imageTensor == None:
            self.imageTensor = ImageUtils.loadImageToTensor(self.maskImageFileName, self.GenJob.ImageSizeX, self.GenJob.ImageSizeY )
        
        print('Adding masked prompt for: "' + self.prompt + '", from ' + str(self))

        # add to prompts list, embed, add to masks, make its index discoverable, ugh.
        self.GenJob.EmbedTextPrompt(self.prompt, self.imageTensor)   


