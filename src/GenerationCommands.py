# classes for adding modifications to images during training
# such as the zoom in stuff
# or masking parts of original image onto currently generated image
# TODO: these are all directly modifying their generation job, i dont like that, but it'll have to be refactored later

from typing import List
from src import GenerationCommand
from src import ImageUtils
from src import MakeCutouts
import numpy as np

import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from PIL import ImageFile, Image, ImageChops
ImageFile.LOAD_TRUNCATED_IMAGES = True



# adds a mask that can be repeatedly pasted on the image
# from the initial source image. kinda preserves a part of the image, and generates on the rest
class OriginalImageMask(GenerationCommand.IGenerationCommand):

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


    def OnExecute(self, iteration: int ):
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
        self.GenJob.SetOptimizer(self.GenJob.optimizerName, self.GenJob.step_size)





# from original inmplementation of image zoom from nerdyRodent
class ImageZoomer(GenerationCommand.IGenerationCommand):
    #fucking python has no maxint to use as a large value, annoying
    def __init__(self, GenJob,zoom_scale: float = 0.99, zoom_shift_x: int = 0, zoom_shift_y: int = 0):
        super().__init__(GenJob)

        ##need to make these configurable
        self.zoom_scale = zoom_scale
        self.zoom_shift_x = zoom_shift_x
        self.zoom_shift_y = zoom_shift_y

    def Initialize(self):
        pass


    def OnExecute(self, iteration: int ):
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
        self.GenJob.optimizer = self.GenJob.SetOptimizer(self.GenJob.optimizerName, self.GenJob.step_size)





# faster tensor based image zoomer, but only zooms in for now
class ImageZoomInFast(GenerationCommand.IGenerationCommand):
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

    def OnExecute(self, iteration: int ):
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
        self.GenJob.SetOptimizer(self.GenJob.optimizerName, self.GenJob.optimizerLearningRate)





class ImageRotate(GenerationCommand.IGenerationCommand):
    def __init__(self, GenJob, angle: int = 1):
        super().__init__(GenJob)

        self.angle = angle

    def Initialize(self):
        pass


    def OnExecute(self, iteration: int ):
        with torch.inference_mode():
            curQuantImg = self.GenJob.GetCurrentImageSynthed()

            #this removes the first dim sized 1 to match the rest
            curQuantImg = torch.squeeze(curQuantImg)

            pil_tensor = TF.rotate(curQuantImg, self.angle)

        # Re-encode original?
        self.GenJob.quantizedImage, *_ = self.GenJob.vqganModel.encode(pil_tensor.to(self.GenJob.vqganDevice).unsqueeze(0) * 2 - 1)
        #self.GenJob.original_quantizedImage = self.GenJob.quantizedImage.detach()
        
        self.GenJob.quantizedImage.requires_grad_(True)
        self.GenJob.SetOptimizer(self.GenJob.optimizerName, self.GenJob.optimizerLearningRate)                



class AddTextPrompt(GenerationCommand.IGenerationCommand):
    def __init__(self, GenJob, prompt:str, clearOtherPrompts:bool = True):
        super().__init__(GenJob)

        self.prompt = prompt
        self.clearOtherPrompts = clearOtherPrompts

    def Initialize(self):
        pass


    def OnExecute(self, iteration: int ):
        if self.clearOtherPrompts == True:
            self.GenJob.embededPrompts = []

        print('Changing prompt to: "' + self.prompt + '", from ' + str(self))
        self.GenJob.EmbedTextPrompt(self.prompt)    


class RemovePrompt(GenerationCommand.IGenerationCommand):
    def __init__(self, GenJob, removeAll:bool = False, removeFirst:bool = False, removeLast:bool = False, removeAtIndex:int = -1):
        super().__init__(GenJob)

        self.removeFirst = removeFirst
        self.removeAll = removeAll
        self.removeLast = removeLast
        self.removeAtIndex = removeAtIndex

    def Initialize(self):
        pass


    def OnExecute(self, iteration: int ):
        if self.removeAll:
            self.GenJob.embededPrompts = []
            print('Removing all prompts, from ' + str(self))
        elif self.removeFirst:
            self.GenJob.embededPrompts.pop(0)
            print('Removing first prompt, from ' + str(self))
        elif self.removeLast:
            self.GenJob.embededPrompts.pop(len(self.GenJob.embededPrompts) - 1)
            print('Removing last prompt, from ' + str(self))
        elif self.removeAtIndex >= 0 and self.removeAtIndex < len(self.GenJob.embededPrompts):
            self.GenJob.embededPrompts.pop( self.removeAtIndex )
            print('Removing prompt at ' + str(self.removeAtIndex) + ', from ' + str(self))
          


class AddTextPromptWithMask(GenerationCommand.IGenerationCommand):
    def __init__(self, GenJob, prompt:str, maskImageFileName:str = None, maskTensor:torch.Tensor = None, dilateMaskAmount:int = 10, blindfold:float = 0.1, cacheImageOnInit:bool = True):
        super().__init__(GenJob)

        self.prompt = prompt        
        self.cacheImageOnInit = cacheImageOnInit        
        self.dilateMaskAmount = dilateMaskAmount
        self.blindfold = blindfold

        self.maskImageFileName = maskImageFileName # filename of mask, if we load off the HD
        self.sourceMaskTensor = maskTensor # if we have a tensor we would liek to use as a mask, pass that in

        self.imageTensor:torch.Tensor = None # the mask tensor we use for the prompt

    def Initialize(self):
        with torch.inference_mode():
            # load image into tensor? or we can do it when its time to add the mod, memory now vs. performance later...
            if self.cacheImageOnInit:
                #cache the image as a tensor here
                if self.sourceMaskTensor == None:
                    self.sourceMaskTensor = ImageUtils.loadImageToTensor(self.maskImageFileName) #, self.GenJob.ImageSizeX, self.GenJob.ImageSizeY )

                self.imageTensor = self.SetupMask(self.sourceMaskTensor)
                del self.sourceMaskTensor


    def SetupMask(self, tensor:torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            #dialate
            tensor = tensor.unsqueeze(0).to(self.GenJob.vqganDevice)

            if self.dilateMaskAmount:                
                struct_ele = torch.FloatTensor(1, 1,self.dilateMaskAmount,self.dilateMaskAmount).fill_(1).to(self.GenJob.vqganDevice)
                tensor = F.conv2d(tensor,struct_ele,padding='same')                

            #resize masks to output size
            tensor = F.interpolate(tensor,(self.GenJob.ImageSizeX, self.GenJob.ImageSizeY ) )

            tensor = tensor.squeeze(0)

            #make binary
            tensor[tensor>0.1]=1

            return tensor.to(self.GenJob.vqganDevice)

    def OnExecute(self, iteration: int ):
        if self.imageTensor == None:
            if self.sourceMaskTensor == None:
                self.sourceMaskTensor = ImageUtils.loadImageToTensor(self.maskImageFileName) #, self.GenJob.ImageSizeX, self.GenJob.ImageSizeY )

            self.imageTensor = self.SetupMask(self.sourceMaskTensor)
            del self.sourceMaskTensor
        
        print('Adding masked prompt for: "' + self.prompt + '", from ' + str(self))

        if self.blindfold != 0.0 and self.GenJob.blur_conv == None:
            #Set up blur used in blindfolding
            k=13
            blur_conv = torch.nn.Conv2d(3,3,k,1,'same',bias=False,padding_mode='reflect',groups=3)
            for param in blur_conv.parameters():
                param.requires_grad = False
            blur_conv.weight[:] = 1/(k**2)

            self.GenJob.blur_conv = blur_conv.to(self.GenJob.vqganDevice)

        # add to prompts list, embed, add to masks, make its index discoverable, ugh.
        self.GenJob.EmbedMaskedPrompt(self.prompt, self.imageTensor, self.blindfold)   


# sets the optimiser used
# see GenerationJob.py, def get_optimizer, to see various optimisers and learnign rates
class SetOptimiser(GenerationCommand.IGenerationCommand):
    def __init__(self, GenJob, optimizerName:str = "Adam", learningRate:float = 0.1):
        super().__init__(GenJob)

        self.optimizerName = optimizerName
        self.learningRate = learningRate

    def Initialize(self):
        pass


    def OnExecute(self, iteration: int ):
        self.GenJob.SetOptimizer(self.optimizerName, self.learningRate)  


# cut method to use, see MakeCutouts.py for various cutout methods
class SetCutMethod(GenerationCommand.IGenerationCommand):
    def __init__(self, GenJob, cut_method:str = "latest", cutNum:int = 32, cutSize:List[int] = [0,0], cutPow:float = 1.0, augments:list = [], useKorniaAugments:bool = True):
        super().__init__(GenJob)

        self.cut_method = cut_method
        self.cutNum = cutNum
        self.cutSize = cutSize
        self.cutPow = cutPow
        self.augments = augments
        self.useKorniaAugments = useKorniaAugments

    def Initialize(self):
        pass


    def OnExecute(self, iteration: int ):
        self.GenJob.SetCutMethod( self.cut_method, self.cutNum, self.cutSize, self.cutPow, self.augments, self.useKorniaAugments )
