import time
import logging
from diffusers import (
    BitsAndBytesConfig, 
    SD3Transformer2DModel, 
    AutoencoderKL, 
    FlowMatchEulerDiscreteScheduler
)
from diffusers.models import SD3ControlNetModel, SD3MultiControlNetModel
from transformers import (
    CLIPTokenizer, 
    T5TokenizerFast,
    DPTImageProcessor, 
    DPTForDepthEstimation,
    CLIPTextModelWithProjection, 
    T5EncoderModel,
)
from .SD35ControlNetPipelineWithCannyFix import SD35ControlNetPipelineWithCannyFix


def load_depth_processor(config, logger):
    """Load depth estimation models"""
    logger.info("Loading Depth estimator...")
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
    
    logger.info(f"Depth processor loading took {time.time() - start_time:.4f} seconds")
    return depth_estimator, feature_extractor


def load_pipeline(config, logger):
    """Load all models and create the pipeline"""
    logger.info("Loading models...")
    start_time = time.time()
    
    # Load ControlNet
    logger.info("Loading Canny ControlNet model...")
    canny_controlnet = SD3ControlNetModel.from_pretrained(
        config.canny_controlnet_path, 
        torch_dtype=config.torch_dtype,
        local_files_only=config.local_files_only,
    ).to(config.device) 
    # MODAL - you might be able to move all this .to stuff
    # later on and snapshot memory.

    logger.info("Loading Depth ControlNet model...")
    depth_controlnet = SD3ControlNetModel.from_pretrained(
        config.depth_controlnet_path, 
        torch_dtype=config.torch_dtype,
        local_files_only=config.local_files_only,
    ).to(config.device)

    logger.info("Loading Blur ControlNet model...")
    blur_controlnet = SD3ControlNetModel.from_pretrained(
        config.blur_controlnet_path, 
        torch_dtype=config.torch_dtype,
        local_files_only=config.local_files_only,
    ).to(config.device)
    
    multi_controlnet = SD3MultiControlNetModel([canny_controlnet, depth_controlnet, blur_controlnet])
    
    # Load VAE
    logger.info("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        config.model_repo,
        subfolder="vae",
        torch_dtype=config.torch_dtype,
        cache_dir=config.cache_dir,
        local_files_only=config.local_files_only,
    ).to(config.device)
    
    # Load Transformer (with optional 4-bit quantization)
    logger.info("Loading Transformer...")
    if config.use_4bit_quantization:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=config.torch_dtype
        )
        transformer = SD3Transformer2DModel.from_pretrained(
            config.model_repo,
            subfolder="transformer",
            quantization_config=nf4_config,
            torch_dtype=config.torch_dtype,
            cache_dir=config.cache_dir,
            local_files_only=config.local_files_only,
        )
    else:
        transformer = SD3Transformer2DModel.from_pretrained(
            config.model_repo,
            subfolder="transformer",
            torch_dtype=config.torch_dtype,
            cache_dir=config.cache_dir,
            local_files_only=config.local_files_only,
        ).to(config.device)
    
    # Load text encoders
    logger.info("Loading text encoders...")
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        config.model_repo,
        subfolder="text_encoder",
        torch_dtype=config.torch_dtype,
        cache_dir=config.cache_dir,
        local_files_only=config.local_files_only,
    ).to(config.device)
    
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        config.model_repo,
        subfolder="text_encoder_2",
        torch_dtype=config.torch_dtype,
        cache_dir=config.cache_dir,
        local_files_only=config.local_files_only,
    ).to(config.device)
    
    text_encoder_3 = T5EncoderModel.from_pretrained(
        config.model_repo,
        subfolder="text_encoder_3",
        torch_dtype=config.torch_dtype,
        cache_dir=config.cache_dir,
        local_files_only=config.local_files_only,
    ).to(config.device)
    
    # Load tokenizers
    logger.info("Loading tokenizers...")
    tokenizer = CLIPTokenizer.from_pretrained(
        config.model_repo,
        subfolder="tokenizer",
        cache_dir=config.cache_dir,
        local_files_only=config.local_files_only,
    )
    
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        config.model_repo,
        subfolder="tokenizer_2",
        cache_dir=config.cache_dir,
        local_files_only=config.local_files_only,
    )
    
    tokenizer_3 = T5TokenizerFast.from_pretrained(
        config.model_repo,
        subfolder="tokenizer_3",
        cache_dir=config.cache_dir,
        local_files_only=config.local_files_only,
    )
    
    # Load scheduler
    logger.info("Loading scheduler...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        config.model_repo,
        subfolder="scheduler",
        cache_dir=config.cache_dir,
        local_files_only=config.local_files_only,
    )
    
    # Assemble pipeline with custom canny fix
    logger.info("Assembling Stable Diffusion pipeline with canny fix...")
    pipe = SD35ControlNetPipelineWithCannyFix.from_pretrained(
        config.model_repo, 
        controlnet=multi_controlnet, 
        torch_dtype=config.torch_dtype,
        cache_dir=config.cache_dir,
        local_files_only=config.local_files_only,
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        text_encoder_3=text_encoder_3,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        tokenizer_3=tokenizer_3,
        transformer=transformer,
        scheduler=scheduler,
    )
    
    pipe.to(config.device)
    
    logger.info(f"Pipeline loading took {time.time() - start_time:.4f} seconds")
    return pipe