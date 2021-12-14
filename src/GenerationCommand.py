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


class IGenerationCommand(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'OnExecute') and 
                callable(subclass.OnExecute) or 
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
    def OnExecute(self, iteration: int ):
        raise NotImplementedError



class GenerationCommandContainer:
    def __init__(self, mod:IGenerationCommand, startIt: int = 0, endIt: int = 9999999999, freq: int = 10):
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

    def OnExecute(self, iteration: int ):
        self.mod.OnExecute(iteration)