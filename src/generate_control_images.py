import logging
import time
import numpy as np
import torch
from PIL import Image
import cv2
import torchvision.transforms.functional as F
import torchvision.transforms as transforms


def generate_depth_map(image, depth_estimator, feature_extractor, config, logger):
    """Generate depth map from input image based on model type"""
    if config.depth_model_type == "dpt":
        return generate_dpt_depth_map(image, depth_estimator, feature_extractor, config, logger)
    elif config.depth_model_type == "depth_anything_v2":
        return generate_depth_anything_v2_map(image, depth_estimator, feature_extractor, config, logger)
    else:
        raise ValueError(f"Unknown depth model type: {config.depth_model_type}")


def generate_dpt_depth_map(image, depth_estimator, feature_extractor, config, logger):
    """Generate depth map from input image using DPT"""
    logger.info("Generating depth map with DPT...")
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
    
    logger.info(f"DPT depth map generation took {time.time() - start_time:.4f} seconds")
    
    return depth_image


def generate_depth_anything_v2_map(image, depth_estimator, feature_extractor, config, logger):
    """Generate depth map from input image using Depth Anything V2"""
    logger.info("Generating depth map with Depth Anything V2...")
    start_time = time.time()
    
    # Process image for depth estimation
    inputs = feature_extractor(images=image, return_tensors="pt")
    
    # Move inputs to device
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(config.device)
    
    with torch.no_grad(), torch.autocast(config.device):
        outputs = depth_estimator(**inputs)
        depth_map = outputs.predicted_depth
    
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
    
    logger.info(f"Depth Anything V2 depth map generation took {time.time() - start_time:.4f} seconds")
    
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
