from src import Hallucinator
from src import GenerateJob

###########################################
#
# Some helper methods for creating isntances of hallucinator from command line args
# and some other default behaviors so we can quickyl make scripts
#
###########################################


# all command line args pushed to class. clean this up once global use of argparse values is cleaned up

def CreateHallucinatorFromArgParse( args ) -> Hallucinator.Hallucinator:
    hallucinatorInst = Hallucinator.Hallucinator( clipModel = args.clip_model,vqgan_config_path=args.vqgan_config, 
                                              vqgan_checkpoint_path=args.vqgan_checkpoint, use_mixed_precision=args.use_mixed_precision,
                                              clip_cpu=args.clip_cpu, cuda_device=args.cuda_device, anomaly_checker = args.anomaly_checker,
                                              deterministic = args.deterministic, log_clip = args.log_clip, log_clip_oneshot = args.log_clip_oneshot, 
                                              log_mem = args.log_mem, display_freq = args.display_freq )
    return hallucinatorInst




def CreateGenerationJobFromArgParse( hallucinatorInst:Hallucinator.Hallucinator, args ) -> GenerateJob.GenerationJob:    
    genJob = GenerateJob.GenerationJob( hallucinatorInst, cut_method = args.cut_method, totalIterations = args.max_iterations, prompts = args.prompts,
                                        image_prompts = args.image_prompts, startingImage = args.init_image, imageSizeXY = args.size, 
                                        cutNum=args.cutn, cutSize=args.cut_size, cutPow=args.cut_pow, augments=args.augments,
                                        optimizerName=args.optimizer,init_weight=args.init_weight, init_noise=args.init_noise,
                                        noise_prompt_seeds=args.noise_prompt_seeds, noise_prompt_weights=args.noise_prompt_weights, prompt_frequency=args.prompt_frequency,
                                        deterministic = args.deterministic, outputDir = args.output_dir, outputFilename = args.output, save_freq = args.save_freq,
                                        save_seq = args.save_seq, save_best = args.save_best)
    return genJob

