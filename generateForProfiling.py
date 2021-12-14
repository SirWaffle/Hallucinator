#######################################################################################
# preserves old function of running from command line args, and generating one image
#######################################################################################

import sys
import gc

sys.path.append('src')

from src import CmdLineArgs
CmdLineArgs.init()

from src import Hallucinator
from src import HallucinatorHelpers
from src import GenerationCommands
from src import GenerateJob
from src import ProfilerHelper

from PIL import ImageFile, PngImagePlugin
ImageFile.LOAD_TRUNCATED_IMAGES = True

#'cumulative'
#'tottime'
@ProfilerHelper.profile(sort_by='cumulative', lines_to_print=80, strip_dirs=False)
def Hallucinate(h, j):
    # process the whole job in one go
    h.ProcessJobFull( j )

def Run():
    #################################
    # start actually doing stuff here
    #################################

    # create the hallucinator class from commandline args
    hallucinatorInst = HallucinatorHelpers.CreateHallucinatorFromArgParse( CmdLineArgs.args )

    # create a job from the same argparse args
    genJob = HallucinatorHelpers.CreateGenerationJobFromArgParse( hallucinatorInst, CmdLineArgs.args )

    # flush, clean, go
    sys.stdout.flush()
    gc.collect()

    try:
        Hallucinate( hallucinatorInst, genJob)

    except KeyboardInterrupt:
        pass




########
# kick off Run()
########

Run()
