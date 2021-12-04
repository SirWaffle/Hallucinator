#######################################################################################
# preserves old function of running from command line args, and generating one image
#######################################################################################

import sys
import gc

#sys.path.append('src')

from src import CmdLineArgs
CmdLineArgs.init()

from src import Hallucinator
from src import HallucinatorHelpers
from src import GenerationMods
from src import GenerateJob

from PIL import ImageFile, PngImagePlugin
ImageFile.LOAD_TRUNCATED_IMAGES = True


#################################
# start actually doing stuff here
#################################

# create the hallucinator class from commandline args
hallucinatorInst = HallucinatorHelpers.CreateHallucinatorFromArgParse( CmdLineArgs.args )
hallucinatorInst.Initialize()


# create a job from the same argparse args
genJob = HallucinatorHelpers.CreateGenerationJobFromArgParse( hallucinatorInst, CmdLineArgs.args )
genJob.Initialize()

# write out the input noise...
info = PngImagePlugin.PngInfo()
info.add_text('comment', f'hallucinator prompt: {CmdLineArgs.args.prompts}')
genJob.SaveCurrentImage( str(0).zfill(5) + '_seed_', info)

# flush, clean, go
sys.stdout.flush()
gc.collect()

try:
    # process the whole job in one go
    hallucinatorInst.ProcessJobFull( genJob )

    # Save final image
    genJob.SaveCurrentImage()

except KeyboardInterrupt:
    pass

