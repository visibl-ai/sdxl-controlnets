import time
import logging
from diffusers import (
    AutoencoderKL, 
    StableDiffusionXLControlNetImg2ImgPipeline,
)
from diffusers.models import ControlNetModel, MultiControlNetModel
from transformers import (
    #CLIPTokenizer, 
    #T5TokenizerFast,
    DPTImageProcessor, 
    DPTForDepthEstimation,
    #CLIPTextModelWithProjection, 
    #T5EncoderModel,
    AutoModelForDepthEstimation,
    AutoImageProcessor,
)


def load_depth_processor(config, logger):
    """Load depth estimation models based on config"""
    if config.depth_model_type == "dpt":
        return load_dpt_depth_processor(config, logger)
    elif config.depth_model_type == "depth_anything_v2":
        return load_depth_anything_v2_processor(config, logger)
    else:
        raise ValueError(f"Unknown depth model type: {config.depth_model_type}")


def load_dpt_depth_processor(config, logger):
    """Load DPT depth estimation models"""
    logger.info("Loading DPT Depth estimator...")
    start_time = time.time()
    
    depth_estimator = DPTForDepthEstimation.from_pretrained(
        config.depth_model,
        cache_dir=config.cache_dir,
        torch_dtype=config.torch_dtype,
        local_files_only=config.local_files_only,
    ).to(config.device)
    
    feature_extractor = DPTImageProcessor.from_pretrained(
        config.depth_model,
        cache_dir=config.cache_dir,
        torch_dtype=config.torch_dtype,
        local_files_only=config.local_files_only
    )
    
    logger.info(f"DPT Depth processor loading took {time.time() - start_time:.4f} seconds")
    return depth_estimator, feature_extractor


def load_depth_anything_v2_processor(config, logger):
    """Load Depth Anything V2 depth estimation models"""
    logger.info("Loading Depth Anything V2 estimator...")
    start_time = time.time()
    
    depth_estimator = AutoModelForDepthEstimation.from_pretrained(
        config.depth_anything_model,
        cache_dir=config.cache_dir,
        torch_dtype=config.torch_dtype,
        local_files_only=config.local_files_only,
    ).to(config.device)
    
    feature_extractor = AutoImageProcessor.from_pretrained(
        config.depth_anything_model,
        cache_dir=config.cache_dir,
        local_files_only=config.local_files_only
    )
    
    logger.info(f"Depth Anything V2 processor loading took {time.time() - start_time:.4f} seconds")
    return depth_estimator, feature_extractor


def load_pipeline(config, logger):
    """Load all models and create the pipeline"""
    logger.info("Loading models...")
    start_time = time.time()
    
    # Load ControlNet
    logger.info("Loading Canny ControlNet model...")
    canny_controlnet = ControlNetModel.from_pretrained(
        config.canny_controlnet_path, 
        torch_dtype=config.torch_dtype,
        local_files_only=config.local_files_only,
    ).to(config.device) 
    # MODAL - you might be able to move all this .to stuff
    # later on and snapshot memory.

    logger.info("Loading Depth ControlNet model...")
    depth_controlnet = ControlNetModel.from_pretrained(
        config.depth_controlnet_path, 
        torch_dtype=config.torch_dtype,
        local_files_only=config.local_files_only,
    ).to(config.device)

    # logger.info("Loading Blur ControlNet model...")
    # blur_controlnet = SD3ControlNetModel.from_pretrained(
    #     config.blur_controlnet_path, 
    #     torch_dtype=config.torch_dtype,
    #     local_files_only=config.local_files_only,
    # ).to(config.device)
    
    multi_controlnet = MultiControlNetModel([canny_controlnet, depth_controlnet])
    
    logger.info("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=config.torch_dtype,
        cache_dir=config.cache_dir,
        local_files_only=config.local_files_only,
    ).to(config.device)
    # Assemble pipeline with custom canny fix
    logger.info("Assembling Stable Diffusion pipeline with canny fix...")
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        config.model_repo, 
        controlnet=multi_controlnet, 
        torch_dtype=config.torch_dtype,
        cache_dir=config.cache_dir,
        local_files_only=config.local_files_only,
        vae=vae,
        use_safetensors=True,
    )
    # MODAL - Maybe you can snapshot memory here?
    pipe.to(config.device) 
    logger.info(f"Pipeline loading took {time.time() - start_time:.4f} seconds")
    return pipe