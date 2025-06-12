import logging
import time
import numpy as np
import torch
from PIL import Image
import cv2
import torchvision.transforms.functional as F
import torchvision.transforms as transforms


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