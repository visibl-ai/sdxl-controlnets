import logging
import time
import os

# Track total import time
total_import_start = time.time()

# Time each import
import_start = time.time()
import numpy as np
numpy_import_time = time.time() - import_start

import_start = time.time()
import torch
torch_import_time = time.time() - import_start

import_start = time.time()
from PIL import Image
pil_import_time = time.time() - import_start

# Diffusers imports
import_start = time.time()
from diffusers.utils import load_image
from diffusers import (
    StableDiffusion3ControlNetPipeline, 
    BitsAndBytesConfig, 
    SD3Transformer2DModel, 
    AutoencoderKL, 
    FlowMatchEulerDiscreteScheduler
)
from diffusers.models import (
    SD3ControlNetModel,
    SD3MultiControlNetModel,
)
diffusers_import_time = time.time() - import_start

# Transformers imports
import_start = time.time()
from transformers import (
    CLIPTokenizer, 
    T5TokenizerFast,
    DPTImageProcessor, 
    DPTForDepthEstimation,
    CLIPTextModelWithProjection, 
    T5EncoderModel,
)
transformers_import_time = time.time() - import_start

# OpenCV import
import_start = time.time()
import cv2
cv2_import_time = time.time() - import_start

import_start = time.time()
from huggingface_hub import login
hf_hub_import_time = time.time() - import_start

total_import_time = time.time() - total_import_start


class Config:
    """Configuration for SD3.5 ControlNet pipeline"""
    # Model paths
    model_repo = "stabilityai/stable-diffusion-3.5-large"
    ## need to convert to diffusers format. 
    ## https://github.com/huggingface/diffusers/blob/6c7fad7ec8b2417c92326804e1751658874fd43b/scripts/convert_sd3_controlnet_to_diffusers.py#L2
    ## python scripts/convert_sd3_controlnet_to_diffusers.py --checkpoint_path "../sd3.5/models/sd3.5_large_controlnet_depth.safetensors" --output_path ../sd3.5/models/sd3.5_large_controlnet_depth_diffusers
    depth_controlnet_path = "/workspace/sd3.5/models/sd3.5_large_controlnet_depth_diffusers"
    canny_controlnet_path = "/workspace/sd3.5/models/sd3.5_large_controlnet_canny_diffusers"
    depth_model = "Intel/dpt-hybrid-midas"
    
    # Cache and environment
    cache_dir = "./cache"
    device = "cuda"
    torch_dtype = torch.bfloat16
    local_files_only = True
    offline_mode = True
    
    # Input/Output paths
    input_image = "./inputs/square.png"
    output_dir = "outputs"
    depth_output = "outputs/diffusers_depth_control.png"
    canny_output = "outputs/diffusers_canny_control.png"
    final_output = "outputs/diffusers_output.png"
    
    # Generation parameters
    prompt = "studio ghibli style"
    negative_prompt = "low quality, incomplete, blurred"
    height = 1024
    width = 1024
    num_inference_steps = 60  # SD3.5 ControlNet recommended
    guidance_scale = 3.5      # SD3.5 ControlNet recommended (lower than default)
    depth_controlnet_conditioning_scale = 0.85
    canny_controlnet_conditioning_scale = 0.85
    depth_control_guidance_start = 0.0
    canny_control_guidance_start = 0.08
    depth_control_guidance_end = 0.15
    canny_control_guidance_end = 1.0
    seed = None  # Set to specific value for reproducibility
    
    # Canny edge detection parameters
    canny_low_threshold = 100
    canny_high_threshold = 200
    
    # Quantization (optional)
    use_4bit_quantization = False
    
    # Logging
    log_level = logging.INFO
    log_format = '%(asctime)s - %(levelname)s - %(message)s'


def setup_environment(config):
    """Setup logging and environment variables"""
    # Configure logging
    logging.basicConfig(level=config.log_level, format=config.log_format)
    logger = logging.getLogger(__name__)
    
    # Log import times
    logger.info("=== Import times ===")
    logger.info(f"numpy import took {numpy_import_time:.4f} seconds")
    logger.info(f"torch import took {torch_import_time:.4f} seconds")
    logger.info(f"PIL import took {pil_import_time:.4f} seconds")
    logger.info(f"diffusers import took {diffusers_import_time:.4f} seconds")
    logger.info(f"transformers import took {transformers_import_time:.4f} seconds")
    logger.info(f"cv2 import took {cv2_import_time:.4f} seconds")
    logger.info(f"huggingface_hub import took {hf_hub_import_time:.4f} seconds")
    logger.info(f"Total import time: {total_import_time:.4f} seconds")
    
    # Set cache directories
    os.environ["HF_HOME"] = config.cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = "hub"
    os.environ["TRANSFORMERS_CACHE"] = "transformers"
    os.environ["HF_DATASETS_CACHE"] = "datasets"
    
    if config.offline_mode:
        os.environ["HF_HUB_OFFLINE"] = "1"
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    logger.info(f"Using device: {config.device}")
    logger.info(f"Model repo: {config.model_repo}")
    logger.info(f"ControlNet path: {config.depth_controlnet_path}")
    logger.info(f"Cache directory: {config.cache_dir}")
    logger.info(f"Torch dtype: {config.torch_dtype}")
    
    return logger


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


def generate_depth_map(image, depth_estimator, feature_extractor, config, logger):
    """Generate depth map from input image"""
    logger.info("Generating depth map...")
    start_time = time.time()
    
    # Process image for depth estimation
    depth_image = feature_extractor(images=image, return_tensors="pt").pixel_values.to(config.device)
    
    with torch.no_grad(), torch.autocast(config.device):
        depth_map = depth_estimator(depth_image).predicted_depth
    
    # Normalize and resize depth map
    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(config.height, config.width),
        mode="bicubic",
        align_corners=False,
    )
    
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    
    # Convert to PIL Image
    depth_image = torch.cat([depth_map] * 3, dim=1)
    depth_image = depth_image.permute(0, 2, 3, 1).cpu().numpy()[0]
    depth_image = Image.fromarray((depth_image * 255.0).clip(0, 255).astype(np.uint8))
    
    logger.info(f"Depth map generation took {time.time() - start_time:.4f} seconds")
    
    return depth_image


def generate_canny_map(image, config, logger):
    """Generate canny edge map from input image"""
    logger.info("Generating canny edge detection...")
    start_time = time.time()
    
    # Convert to numpy array
    logger.info("Converting image to numpy array...")
    np_conversion_start = time.time()
    np_image = np.array(image)
    np_conversion_time = time.time() - np_conversion_start
    logger.info(f"Numpy conversion took {np_conversion_time:.4f} seconds")
    
    # Generate canny edge detection
    logger.info(f"Generating Canny edge detection (thresholds: {config.canny_low_threshold}, {config.canny_high_threshold})...")
    canny_start = time.time()
    np_image = cv2.Canny(np_image, config.canny_low_threshold, config.canny_high_threshold)
    np_image = np_image[:, :, None]
    np_image = np.concatenate([np_image, np_image, np_image], axis=2)
    canny_image = Image.fromarray(np_image)
    canny_time = time.time() - canny_start
    logger.info(f"Canny edge detection took {canny_time:.4f} seconds")
    
    # Resize if needed
    if canny_image.size != (config.width, config.height):
        canny_image = canny_image.resize((config.width, config.height), Image.LANCZOS)
    
    logger.info(f"Canny map generation took {time.time() - start_time:.4f} seconds")
    
    return canny_image


def load_pipeline(config, logger):
    """Load all models and create the pipeline"""
    logger.info("Loading models...")
    start_time = time.time()
    
    # Load ControlNet
    logger.info("Loading Depth ControlNet model...")
    depth_controlnet = SD3ControlNetModel.from_pretrained(
        config.depth_controlnet_path, 
        torch_dtype=config.torch_dtype,
        local_files_only=config.local_files_only,
    ).to(config.device)
    
    logger.info("Loading Canny ControlNet model...")
    canny_controlnet = SD3ControlNetModel.from_pretrained(
        config.canny_controlnet_path, 
        torch_dtype=config.torch_dtype,
        local_files_only=config.local_files_only,
    ).to(config.device)

    multi_controlnet = SD3MultiControlNetModel([depth_controlnet, canny_controlnet])
    
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
    
    # Assemble pipeline
    logger.info("Assembling Stable Diffusion pipeline...")
    pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
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


def generate_image(pipeline, depth_image, canny_image, config, logger):
    """Generate image using the pipeline"""
    logger.info("Starting image generation...")
    start_time = time.time()
    
    # Set up generator for reproducibility
    generator = None
    if config.seed is not None:
        generator = torch.Generator(device=config.device).manual_seed(config.seed)
    
    # Generate image
    result = pipeline(
        config.prompt,
        negative_prompt=config.negative_prompt,
        control_image=[depth_image, canny_image],
        controlnet_conditioning_scale=[config.depth_controlnet_conditioning_scale, config.canny_controlnet_conditioning_scale],
        generator=generator,
        height=config.height, 
        width=config.width,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        control_guidance_start=[config.depth_control_guidance_start, config.canny_control_guidance_start],  # Depth starts at 0%, Canny at 8%
        control_guidance_end=[config.depth_control_guidance_end, config.canny_control_guidance_end],    # Depth ends at 15%, Canny at 100%
    )
    
    image = result.images[0]
    logger.info(f"Image generation took {time.time() - start_time:.4f} seconds")
    
    return image


def main():
    """Main execution function"""
    # Track total execution time
    total_start = time.time()
    
    # Initialize configuration
    config = Config()
    
    # Setup environment and logging
    logger = setup_environment(config)
    
    try:
        # Load input image once
        logger.info("Loading input image...")
        image_load_start = time.time()
        input_image = load_image(config.input_image)
        logger.info(f"Image loading took {time.time() - image_load_start:.4f} seconds")
        
        # Load depth processor
        depth_estimator, feature_extractor = load_depth_processor(config, logger)
        
        # Generate depth map
        depth_image = generate_depth_map(
            input_image, 
            depth_estimator, 
            feature_extractor, 
            config, 
            logger
        )
        
        # Save depth map
        logger.info("Saving depth map...")
        depth_image.save(config.depth_output)
        
        # Generate canny map
        canny_image = generate_canny_map(input_image, config, logger)
        
        # Save canny map
        logger.info("Saving canny map...")
        canny_output_path = os.path.join(config.canny_output)
        canny_image.save(canny_output_path)
        
        # Load pipeline
        pipeline = load_pipeline(config, logger)
        
        # Generate image using both depth and canny control
        generated_image = generate_image(pipeline, depth_image, canny_image, config, logger)
        
        # Save generated image
        logger.info("Saving generated image...")
        generated_image.save(config.final_output)
        
        # Summary
        total_time = time.time() - total_start
        logger.info(f"Total execution time: {total_time:.4f} seconds")
        logger.info(f"Output saved to: {config.final_output}")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()