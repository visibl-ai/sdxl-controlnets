import logging
from diffusers import StableDiffusion3ControlNetPipeline
from diffusers.models import SD3ControlNetModel, SD3MultiControlNetModel


class SD35ControlNetPipelineWithCannyFix(StableDiffusion3ControlNetPipeline):
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