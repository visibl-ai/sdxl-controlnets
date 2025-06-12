import time
import logging
import torch
from PIL import Image
from .generate_control_images import generate_depth_map, generate_canny_map, generate_blur_map


def generate_image(pipeline, depth_image, canny_image, blur_image, config, logger):
    """Generate image using the pipeline"""
    logger.info("Starting image generation with multi-controlnet...")
    start_time = time.time()
    
    # Set up generator for reproducibility
    if config.seed is None:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
        logger.info(f"No seed provided. Using random seed: {seed}")
    else:
        seed = config.seed
        logger.info(f"Using provided seed: {seed}")

    generator = torch.Generator(device=config.device).manual_seed(seed)
    
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