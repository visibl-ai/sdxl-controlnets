import time
import logging
import torch
from PIL import Image
from .generate_control_images import generate_depth_map, generate_canny_map, generate_blur_map
from .image_preprocessing import pre_process_input


def generate_image(pipeline, processed_image, depth_image, canny_image, config, logger):
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
        image=processed_image,
        strength=config.original_strength,
        control_image=[canny_image, depth_image,],#, blur_image],
        controlnet_conditioning_scale=[
            config.canny_controlnet_conditioning_scale, 
            config.depth_controlnet_conditioning_scale,
            #config.blur_controlnet_conditioning_scale
        ],
        generator=generator,
        height=config.height, 
        width=config.width,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        control_guidance_start=[
            config.canny_control_guidance_start, 
            config.depth_control_guidance_start,
            #config.blur_control_guidance_start
        ],
        control_guidance_end=[
            config.canny_control_guidance_end, 
            config.depth_control_guidance_end,
            #config.blur_control_guidance_end
        ],
    )
    
    image = result.images[0]
    logger.info(f"Image generation took {time.time() - start_time:.4f} seconds")
    
    return image


def process_single_generation(pipeline, depth_estimator, feature_extractor, config, logger):
    """Process a single generation with the given configuration"""
    generation_start = time.time()
    # Check local_files_only setting
    if not config.local_files_only:
        logger.warning("\n" + "!"*80)
        logger.warning("local_files_only is set to False - this may cause unnecessary network requests")
        logger.warning("For better performance, set local_files_only: true in config after first run")
        logger.warning("!"*80 + "\n")
    try:
        # Load input image
        logger.info(f"Loading input image: {config.input_image}")
        image_load_start = time.time()
        input_image = Image.open(config.input_image).convert('RGB')
        logger.info(f"Image loading took {time.time() - image_load_start:.4f} seconds")
        
        # Preprocess image to supported dimensions
        logger.info("=== Preprocessing input image ===")
        preprocess_start = time.time()
        target_aspect_ratio = config.get_target_aspect_ratio()
        processed_image, width, height = pre_process_input(
            input_image, 
            target_aspect_ratio=target_aspect_ratio,
            logger=logger
        )
        
        # Update config dimensions
        config.width = width
        config.height = height
        
        # Validate dimensions
        config.validate_dimensions()
        
        logger.info(f"Preprocessing took {time.time() - preprocess_start:.4f} seconds")
        
        # Process control images using preprocessed image
        logger.info("=== Processing control images ===")
        
        # Generate depth map
        depth_image = generate_depth_map(
            processed_image,  # Use preprocessed image
            depth_estimator, 
            feature_extractor, 
            config, 
            logger
        )
        
        # Save depth map
        logger.info("Saving depth map...")
        depth_image.save(config.depth_output)
        
        # Generate canny map
        canny_image = generate_canny_map(processed_image, config, logger)  # Use preprocessed image
        
        # Save canny map
        logger.info("Saving canny map...")
        canny_image.save(config.canny_output)
        
        # Generate blur map
        blur_image = generate_blur_map(processed_image, config, logger)  # Use preprocessed image
        
        # Save blur map
        logger.info("Saving blur map...")
        blur_image.save(config.blur_output)
        
        # Generate final image
        logger.info("=== Generating final image ===")
        # The custom pipeline will handle the special preprocessing internally
        generated_image = generate_image(pipeline, processed_image, depth_image, canny_image, config, logger)
        
        # Save generated image
        logger.info("Saving generated image...")
        generated_image.save(config.final_output)
        
        logger.info(f"Generation completed in {time.time() - generation_start:.4f} seconds")
        logger.info(f"Output saved to: {config.final_output}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        return False