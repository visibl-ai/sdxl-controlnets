import logging
from PIL import Image
from typing import Tuple, Optional


# Supported dimensions for SD3.5 model
SUPPORTED_DIMENSIONS = {
    "square": (1024, 1024),
    "landscape": (1344, 768),  # 16:9
    "portrait": (768, 1344)    # 9:16
}


def get_aspect_ratio_type(width: int, height: int) -> str:
    """Determine the aspect ratio type from image dimensions."""
    ratio = width / height
    
    # Define thresholds for aspect ratio detection
    if 0.9 <= ratio <= 1.1:
        return "square"
    elif ratio >= 1.5:  # Wider than 3:2
        return "landscape"
    else:
        return "portrait"


def calculate_resize_dimensions(
    input_width: int, 
    input_height: int, 
    target_width: int, 
    target_height: int
) -> Tuple[int, int]:
    """
    Calculate dimensions to resize image while maintaining aspect ratio.
    The image will be resized to fill the target dimensions (may exceed in one dimension).
    """
    input_ratio = input_width / input_height
    target_ratio = target_width / target_height
    
    if input_ratio > target_ratio:
        # Input is wider - match height and let width exceed
        new_height = target_height
        new_width = int(target_height * input_ratio)
    else:
        # Input is taller - match width and let height exceed
        new_width = target_width
        new_height = int(target_width / input_ratio)
    
    return new_width, new_height


def center_crop(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """Center crop an image to target dimensions."""
    width, height = image.size
    
    # Calculate crop coordinates
    left = (width - target_width) // 2
    top = (height - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    
    return image.crop((left, top, right, bottom))


def pre_process_input(
    image: Image.Image,
    target_aspect_ratio: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> Tuple[Image.Image, int, int]:
    """
    Preprocess input image to match supported SD3.5 dimensions.
    
    Args:
        image: Input PIL Image
        target_aspect_ratio: Optional forced aspect ratio ("square", "landscape", "portrait")
        logger: Optional logger instance
    
    Returns:
        Tuple of (processed_image, width, height)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Ensure image is in RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    input_width, input_height = image.size
    logger.info(f"Input image dimensions: {input_width}x{input_height}")
    
    # Determine target aspect ratio
    if target_aspect_ratio and target_aspect_ratio in SUPPORTED_DIMENSIONS:
        aspect_type = target_aspect_ratio
        logger.info(f"Using specified aspect ratio: {aspect_type}")
    else:
        aspect_type = get_aspect_ratio_type(input_width, input_height)
        logger.info(f"Auto-detected aspect ratio type: {aspect_type}")
    
    # Get target dimensions
    target_width, target_height = SUPPORTED_DIMENSIONS[aspect_type]
    logger.info(f"Target dimensions: {target_width}x{target_height}")
    
    # Process based on aspect ratio type
    if aspect_type == "square":
        # For square, center crop to square first, then resize
        crop_size = min(input_width, input_height)
        
        # Center crop to square
        left = (input_width - crop_size) // 2
        top = (input_height - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        
        cropped = image.crop((left, top, right, bottom))
        logger.info(f"Center cropped to square: {crop_size}x{crop_size}")
        
        # Resize to target dimensions
        processed = cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
    else:
        # For landscape/portrait, resize to fill then center crop
        # Calculate resize dimensions to fill the target
        resize_width, resize_height = calculate_resize_dimensions(
            input_width, input_height, target_width, target_height
        )
        logger.info(f"Resizing to: {resize_width}x{resize_height}")
        
        # Resize image
        resized = image.resize((resize_width, resize_height), Image.Resampling.LANCZOS)
        
        # Center crop to exact target dimensions
        processed = center_crop(resized, target_width, target_height)
    
    logger.info(f"Final processed dimensions: {processed.size[0]}x{processed.size[1]}")
    
    return processed, target_width, target_height