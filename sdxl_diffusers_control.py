import logging
import time
import os
import json
import argparse
import sys

# Log start of imports
print("Starting imports...")

# Track total import time
total_import_start = time.time()

# Diffusers imports
import_start = time.time()
from diffusers.utils import load_image
diffusers_import_time = time.time() - import_start

import_start = time.time()
from huggingface_hub import login
hf_hub_import_time = time.time() - import_start

# Import custom modules
import_start = time.time()
from src.config import Config
from src.pipeline import load_depth_processor, load_pipeline
from src.generation import process_single_generation
custom_imports_time = time.time() - import_start

total_import_time = time.time() - total_import_start


def setup_environment(config):
    """Setup logging and environment variables"""
    # Configure logging
    logging.basicConfig(level=config.log_level, format=config.log_format)
    logger = logging.getLogger(__name__)
    
    # Log import times
    logger.info("=== Import times ===")
    logger.info(f"diffusers import took {diffusers_import_time:.4f} seconds")
    logger.info(f"huggingface_hub import took {hf_hub_import_time:.4f} seconds")
    logger.info(f"custom imports took {custom_imports_time:.4f} seconds")
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


def interactive_mode(pipeline, refiner, depth_estimator, feature_extractor, logger):
    """Run in interactive mode with config.json monitoring"""
    config_path = "config.json"
    
    while True:
        try:
            # Try to load config.json
            if os.path.exists(config_path):
                logger.info(f"Loading configuration from {config_path}")
                try:
                    with open(config_path, 'r') as f:
                        config_dict = json.load(f)
                    
                    # Create config with overrides
                    config = Config(config_dict)
                    logger.info("Configuration loaded successfully")
                except Exception as e:
                    logger.error(f"Error loading {config_path}: {str(e)}")
                    logger.info("Using default configuration")
                    config = Config()
            else:
                logger.info(f"{config_path} not found, using default configuration")
                config = Config()
            
            # Process generation
            logger.info("\n=== Starting generation ===")
            success = process_single_generation(pipeline, refiner, depth_estimator, feature_extractor, config, logger)
            
            if success:
                logger.info("\nGeneration complete!")
            else:
                logger.error("\nGeneration failed!")
            
            # Ask user what to do next
            logger.info("\n" + "="*60)
            logger.info("Options:")
            logger.info("  - Modify config.json and press ENTER to generate again")
            logger.info("  - Press Ctrl+C to exit")
            logger.info("="*60)
            
            try:
                input("\nPress ENTER to generate again (or Ctrl+C to exit)...")
            except KeyboardInterrupt:
                logger.info("\nExiting...")
                break
                
        except KeyboardInterrupt:
            logger.info("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Unexpected error in interactive mode: {str(e)}")
            logger.info("Press ENTER to try again or Ctrl+C to exit...")
            try:
                input()
            except KeyboardInterrupt:
                break


def main():
    """Main execution function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='SDXL ControlNet Pipeline')
    parser.add_argument('--config', type=str, help='Path to JSON configuration file containing array of configurations')
    args = parser.parse_args()
    
    # Initialize base configuration for initial setup
    base_config = Config()
    
    # Setup environment and logging with base config
    logger = setup_environment(base_config)
    
    try:
        # Load pipeline and preprocessing models once (before processing any configs)
        logger.info("=== Loading pipeline and preprocessing models ===")
        pipeline, refiner = load_pipeline(base_config, logger)
        depth_estimator, feature_extractor = load_depth_processor(base_config, logger)
        logger.info("All models loaded successfully")
        
        # MODAL - inference starts here. 

        if args.config:
            # Load configurations from JSON file
            logger.info(f"Loading configurations from: {args.config}")
            
            try:
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
                    
                    try:
                        # Create a new config with overrides
                        item_config = Config(config_dict)
                        
                        # Process this configuration
                        if process_single_generation(pipeline, refiner, depth_estimator, feature_extractor, item_config, logger):
                            successful += 1
                        else:
                            failed += 1
                    except Exception as e:
                        logger.error(f"Error processing configuration {i+1}: {str(e)}")
                        logger.error(f"Skipping to next configuration...")
                        failed += 1
                        continue
                
                logger.info(f"\n{'='*60}")
                logger.info(f"Batch processing complete: {successful} successful, {failed} failed")
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON configuration file: {str(e)}")
                raise
            except FileNotFoundError:
                logger.error(f"Configuration file not found: {args.config}")
                raise
            except Exception as e:
                logger.error(f"Error loading configuration file: {str(e)}")
                raise
            
        else:
            # No config file provided - run in interactive mode
            logger.info("No batch configuration file provided, entering interactive mode")
            logger.info("You can create/modify 'config.json' to override default settings")
            logger.info("\nExample config.json:")
            logger.info(json.dumps({
                "prompt": "your prompt here",
                "input_image": "./inputs/your_image.png",
                "guidance_scale": 3.5,
                "num_inference_steps": 60
            }, indent=2))
            
            # Enter interactive mode
            interactive_mode(pipeline, refiner, depth_estimator, feature_extractor, logger)
        
        # Summary
        total_time = time.time() - total_import_start
        logger.info(f"Total execution time: {total_time:.4f} seconds")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()