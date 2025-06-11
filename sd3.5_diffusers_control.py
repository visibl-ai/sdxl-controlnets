import logging
import time
import os
import json
import argparse
import sys

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

# Torchvision import
import_start = time.time()
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
torchvision_import_time = time.time() - import_start

import_start = time.time()
from huggingface_hub import login
hf_hub_import_time = time.time() - import_start

total_import_time = time.time() - total_import_start


class SD3ControlNetPipelineWithCannyFix(StableDiffusion3ControlNetPipeline):
    """
    Custom SD3.5 ControlNet pipeline that properly handles canny preprocessing.
    
    The standard diffusers implementation doesn't apply the special preprocessing
    required for SD3.5 canny controlnets (image * 255 * 0.5 + 0.5).
    
    For multi-controlnet, assumes the order is [canny, depth, blur]
    """
    
    def prepare_control_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance,
        guess_mode,
        is_canny=False,
    ):
        """Prepare a single control image with optional canny preprocessing"""
        # First call the parent's prepare_image method to get standard preprocessing
        image = super().prepare_image(
            image, width, height, batch_size, num_images_per_prompt,
            device, dtype, do_classifier_free_guidance, guess_mode
        )
        
        # Apply special preprocessing only for canny
        logger = logging.getLogger(__name__)
        if is_canny:
            # At this point, image is a tensor normalized to [-1, 1]
            logger.info(f"Applying special SD3.5 canny preprocessing")
            logger.info(f"  Input tensor shape: {image.shape}")
            logger.info(f"  Input tensor range: [{image.min().item():.2f}, {image.max().item():.2f}]")
            
            # Denormalize from [-1, 1] to [0, 1]
            image = (image + 1.0) / 2.0
            
            # Apply canny preprocessing: * 255 * 0.5 + 0.5
            # This maps [0, 1] to [0.5, 128]
            image = image * 255 * 0.5 + 0.5
            
            # Log after preprocessing
            logger.info(f"  Output tensor range: [{image.min().item():.2f}, {image.max().item():.2f}]")
        else:
            logger.info(f"Using standard preprocessing")
        
        return image
    
    def __call__(self, *args, **kwargs):
        """Override to handle multi-controlnet preprocessing correctly"""
        # Extract the control_image argument
        control_image = kwargs.get('control_image', args[11] if len(args) > 11 else None)
        
        # If we have multi-controlnet, we need to handle the preprocessing differently
        if isinstance(self.controlnet, SD3MultiControlNetModel) and control_image is not None:
            # Store the original prepare_image method
            original_prepare_image = self.prepare_image
            
            # Create a custom prepare_image that uses our enhanced version
            def custom_prepare_image(image, width, height, batch_size, num_images_per_prompt, 
                                   device, dtype, do_classifier_free_guidance=False, guess_mode=False):
                # Determine which controlnet we're processing based on position
                for i, img in enumerate(control_image):
                    if img is image:
                        # First controlnet is always canny
                        is_canny = (i == 0)
                        
                        logger = logging.getLogger(__name__)
                        if i == 0:
                            control_type_name = 'canny'
                        elif i == 1:
                            control_type_name = 'depth'
                        else:
                            control_type_name = 'blur'
                        logger.info(f"Processing control image {i} ({control_type_name})")
                        
                        return self.prepare_control_image(
                            image, width, height, batch_size, num_images_per_prompt,
                            device, dtype, do_classifier_free_guidance, guess_mode, is_canny
                        )
                
                # Fallback to standard processing
                return self.prepare_control_image(
                    image, width, height, batch_size, num_images_per_prompt,
                    device, dtype, do_classifier_free_guidance, guess_mode, False
                )
            
            # Temporarily replace prepare_image
            self.prepare_image = custom_prepare_image
            try:
                result = super().__call__(*args, **kwargs)
            finally:
                # Restore original prepare_image
                self.prepare_image = original_prepare_image
            
            return result
        else:
            # Single controlnet case - assume it's canny if using this custom pipeline
            if control_image is not None and isinstance(self.controlnet, SD3ControlNetModel):
                logger = logging.getLogger(__name__)
                logger.info(f"Processing single control image (assuming canny)")
                
                # Store the original prepare_image method
                original_prepare_image = self.prepare_image
                
                # Create a custom prepare_image for single controlnet
                def custom_prepare_image(image, width, height, batch_size, num_images_per_prompt, 
                                       device, dtype, do_classifier_free_guidance=False, guess_mode=False):
                    return self.prepare_control_image(
                        image, width, height, batch_size, num_images_per_prompt,
                        device, dtype, do_classifier_free_guidance, guess_mode, True  # Assume canny
                    )
                
                # Temporarily replace prepare_image
                self.prepare_image = custom_prepare_image
                try:
                    result = super().__call__(*args, **kwargs)
                finally:
                    # Restore original prepare_image
                    self.prepare_image = original_prepare_image
                
                return result
            else:
                # No control image or unrecognized controlnet type
                return super().__call__(*args, **kwargs)


class Config:
    """Configuration for SD3.5 ControlNet pipeline"""
    def __init__(self, json_config=None):
        # Model paths
        self.model_repo = "stabilityai/stable-diffusion-3.5-large"
        ## need to convert to diffusers format. 
        ## https://github.com/huggingface/diffusers/blob/6c7fad7ec8b2417c92326804e1751658874fd43b/scripts/convert_sd3_controlnet_to_diffusers.py#L2
        ## python scripts/convert_sd3_controlnet_to_diffusers.py --checkpoint_path "../sd3.5/models/sd3.5_large_controlnet_depth.safetensors" --output_path ../sd3.5/models/sd3.5_large_controlnet_depth_diffusers
        self.depth_controlnet_path = "/workspace/sd3.5/models/sd3.5_large_controlnet_depth_diffusers"
        self.canny_controlnet_path = "/workspace/sd3.5/models/sd3.5_large_controlnet_canny_diffusers"
        self.blur_controlnet_path = "/workspace/sd3.5/models/sd3.5_large_controlnet_blur_diffusers"
        self.depth_model = "Intel/dpt-hybrid-midas"
        
        # Cache and environment
        self.cache_dir = "./cache"
        self.device = "cuda"
        self.torch_dtype = torch.bfloat16
        self.local_files_only = True
        self.offline_mode = True
        
        # Input/Output paths
        self.input_image = "./inputs/square.png"
        self.output_dir = "outputs"
        self.depth_output = "outputs/diffusers_depth_control.png"
        self.canny_output = "outputs/diffusers_canny_control.png"
        self.blur_output = "outputs/diffusers_blur_control.png"
        self.final_output = "outputs/diffusers_output.png"
        
        # Generation parameters
        self.prompt = "studio ghibli style"
        self.negative_prompt = "low quality, incomplete, blurred, deformed"
        self.height = 1024
        self.width = 1024
        self.num_inference_steps = 60  # SD3.5 ControlNet recommended
        self.guidance_scale = 3.5      # SD3.5 ControlNet recommended (lower than default)
        self.depth_controlnet_conditioning_scale = 0.7
        self.canny_controlnet_conditioning_scale = 0.7
        self.blur_controlnet_conditioning_scale = 0.7
        self.depth_control_guidance_start = 0.0
        self.canny_control_guidance_start = 0.0
        self.blur_control_guidance_start = 0.0
        self.depth_control_guidance_end = 1.0
        self.canny_control_guidance_end = 1.0
        self.blur_control_guidance_end = 1.0
        self.seed = None  # Set to specific value for reproducibility
        
        # Canny edge detection parameters
        self.canny_low_threshold = 50
        self.canny_high_threshold = 200
        
        # Blur parameters
        self.blur_kernel_size = 101  # Must be odd number
        
        # Quantization (optional)
        self.use_4bit_quantization = False
        
        # Logging
        self.log_level = logging.INFO
        self.log_format = '%(asctime)s - %(levelname)s - %(message)s'
        
        # Override with JSON config if provided
        if json_config:
            self.update_from_json(json_config)
    
    def update_from_json(self, json_config):
        """Update configuration from JSON dictionary"""
        logger = logging.getLogger(__name__)
        
        # Handle special cases for data types
        for key, value in json_config.items():
            if hasattr(self, key):
                # Handle torch dtype specially
                if key == 'torch_dtype' and isinstance(value, str):
                    if value == 'bfloat16':
                        self.torch_dtype = torch.bfloat16
                    elif value == 'float16':
                        self.torch_dtype = torch.float16
                    elif value == 'float32':
                        self.torch_dtype = torch.float32
                    else:
                        logger.warning(f"Unknown torch dtype: {value}, keeping default")
                # Handle log level specially
                elif key == 'log_level' and isinstance(value, str):
                    level_map = {
                        'DEBUG': logging.DEBUG,
                        'INFO': logging.INFO,
                        'WARNING': logging.WARNING,
                        'ERROR': logging.ERROR,
                        'CRITICAL': logging.CRITICAL
                    }
                    if value.upper() in level_map:
                        self.log_level = level_map[value.upper()]
                    else:
                        logger.warning(f"Unknown log level: {value}, keeping default")
                else:
                    setattr(self, key, value)
                    logger.info(f"Config override: {key} = {value}")
            else:
                logger.warning(f"Unknown config key in JSON: {key}")
    
    @classmethod
    def from_json_file(cls, json_path):
        """Create Config instance from JSON file"""
        with open(json_path, 'r') as f:
            json_config = json.load(f)
        return cls(json_config)


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
    logger.info(f"torchvision import took {torchvision_import_time:.4f} seconds")
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
    preprocess_start = time.time()
    
    # Convert PIL to tensor then to numpy
    img_tensor = F.to_tensor(image)
    img_np = img_tensor.numpy()
    
    # Convert to grayscale
    img_gray = cv2.cvtColor(img_np.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
    
    # Convert to uint8 for Canny
    img_gray = (img_gray * 255).astype('uint8')
    
    # Apply Canny edge detection
    edges = cv2.Canny(img_gray, config.canny_low_threshold, config.canny_high_threshold)
    
    # Convert back to PIL Image
    edges_pil = Image.fromarray(edges)
    
    # Convert to RGB (Canny outputs single channel)
    edges_rgb = edges_pil.convert('RGB')
    # Log canny image dimensions
    canny_width, canny_height = edges_rgb.size
    logger.info(f"  Canny edge map dimensions: {canny_width}x{canny_height}")
    # Resize if needed
    # if edges_rgb.size != (config.width, config.height):
    #     edges_rgb = edges_rgb.resize((config.width, config.height), Image.LANCZOS)
    
    logger.info(f"Canny preprocessing completed in {time.time() - preprocess_start:.2f}s")
    return edges_rgb


def generate_blur_map(image, config, logger):
    """Generate blur map from input image"""
    logger.info("Generating blur map...")
    blur_start = time.time()
    
    # Convert PIL image to tensor for processing
    image_tensor = F.to_tensor(image)
    
    # Create gaussian blur transform with explicit kernel size
    kernel_size = config.blur_kernel_size
    # Use fixed sigma based on kernel size to ensure consistent blur
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    gaussian_blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    
    # Apply blur to the tensor
    blurred_tensor = gaussian_blur(image_tensor)
    
    # Convert back to PIL Image
    blurred_image = F.to_pil_image(blurred_tensor)
    
    # Log blur image dimensions and debug info
    blur_width, blur_height = blurred_image.size
    logger.info(f"  Blur map dimensions: {blur_width}x{blur_height}")
    logger.info(f"  Blur kernel size: {kernel_size}")
    logger.info(f"  Input image size: {image.size}")
    logger.info(f"  Blur transform: {gaussian_blur}")
    
    logger.info(f"Blur preprocessing completed in {time.time() - blur_start:.2f}s")
    return blurred_image


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
    pipe = SD3ControlNetPipelineWithCannyFix.from_pretrained(
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


def generate_image(pipeline, depth_image, canny_image, blur_image, config, logger):
    """Generate image using the pipeline"""
    logger.info("Starting image generation with multi-controlnet...")
    start_time = time.time()
    
    # Set up generator for reproducibility
    generator = None
    if config.seed is not None:
        generator = torch.Generator(device=config.device).manual_seed(config.seed)
    
    # Generate image - the custom pipeline will handle canny preprocessing
    result = pipeline(
        config.prompt,
        negative_prompt=config.negative_prompt,
        control_image=[canny_image, depth_image, blur_image],
        controlnet_conditioning_scale=[
            config.canny_controlnet_conditioning_scale, 
            config.depth_controlnet_conditioning_scale,
            config.blur_controlnet_conditioning_scale
        ],
        generator=generator,
        height=config.height, 
        width=config.width,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        control_guidance_start=[
            config.canny_control_guidance_start, 
            config.depth_control_guidance_start,
            config.blur_control_guidance_start
        ],
        control_guidance_end=[
            config.canny_control_guidance_end, 
            config.depth_control_guidance_end,
            config.blur_control_guidance_end
        ],
    )
    
    image = result.images[0]
    logger.info(f"Image generation took {time.time() - start_time:.4f} seconds")
    
    return image


def process_single_generation(pipeline, depth_estimator, feature_extractor, config, logger):
    """Process a single generation with the given configuration"""
    generation_start = time.time()
    
    try:
        # Load input image
        logger.info(f"Loading input image: {config.input_image}")
        image_load_start = time.time()
        input_image = Image.open(config.input_image).convert('RGB')
        logger.info(f"Image loading took {time.time() - image_load_start:.4f} seconds")
        
        # Process control images
        logger.info("=== Processing control images ===")
        
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
        canny_image.save(config.canny_output)
        
        # Generate blur map
        blur_image = generate_blur_map(input_image, config, logger)
        
        # Save blur map
        logger.info("Saving blur map...")
        blur_image.save(config.blur_output)
        
        # Generate final image
        logger.info("=== Generating final image ===")
        # The custom pipeline will handle the special preprocessing internally
        generated_image = generate_image(pipeline, depth_image, canny_image, blur_image, config, logger)
        
        # Save generated image
        logger.info("Saving generated image...")
        generated_image.save(config.final_output)
        
        logger.info(f"Generation completed in {time.time() - generation_start:.4f} seconds")
        logger.info(f"Output saved to: {config.final_output}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        return False


def main():
    """Main execution function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='SD3.5 ControlNet Pipeline')
    parser.add_argument('--config', type=str, help='Path to JSON configuration file containing array of configurations')
    args = parser.parse_args()
    
    # Initialize base configuration for initial setup
    base_config = Config()
    
    # Setup environment and logging with base config
    logger = setup_environment(base_config)
    
    try:
        # Load pipeline and preprocessing models once (before processing any configs)
        logger.info("=== Loading pipeline and preprocessing models ===")
        pipeline = load_pipeline(base_config, logger)
        depth_estimator, feature_extractor = load_depth_processor(base_config, logger)
        logger.info("All models loaded successfully")
        
        # MODAL - inference starts here. 

        if args.config:
            # Load configurations from JSON file
            logger.info(f"Loading configurations from: {args.config}")
            
            with open(args.config, 'r') as f:
                configs = json.load(f)
            
            if not isinstance(configs, list):
                raise ValueError("Configuration file must contain a JSON array of configurations")
            
            if len(configs) == 0:
                raise ValueError("Configuration array is empty")
            
            logger.info(f"Found {len(configs)} configurations to process")
            
            # Process each configuration
            successful = 0
            failed = 0
            
            for i, config_dict in enumerate(configs):
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing configuration {i+1}/{len(configs)}")
                
                # Create a new config with overrides
                item_config = Config(config_dict)
                
                # Process this configuration
                if process_single_generation(pipeline, depth_estimator, feature_extractor, item_config, logger):
                    successful += 1
                else:
                    failed += 1
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Batch processing complete: {successful} successful, {failed} failed")
            
        else:
            # No config file provided - run with defaults
            logger.info("No configuration file provided, running with default settings")
            
            # Process single generation with defaults
            process_single_generation(pipeline, depth_estimator, feature_extractor, base_config, logger)
        
        # Summary
        total_time = time.time() - total_import_start
        logger.info(f"Total execution time: {total_time:.4f} seconds")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()