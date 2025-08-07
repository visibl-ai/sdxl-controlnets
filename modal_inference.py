import logging
import os
import time
import tempfile
from urllib.parse import urlparse, urlunparse

import modal
import requests
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import modal_settings, settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_DIR = "/cache"
RESULTS_DIR = "/results"  # Define results directory as absolute path
MINUTES = 60

from modal import App

app = App(modal_settings.inference_app_id)

# Create the base image and include local dependencies
image = (
    modal.Image.debian_slim(python_version="3.11.6")
    .run_commands(
        "apt-get update",
        "apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6",
    )
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir("src", "/root/src", copy=True, ignore=["__pycache__", "*.pyc"])
    .add_local_file("config.py", "/root/config.py", copy=True)
    .add_local_file("sdxl_diffusers_control.py", "/root/sdxl_diffusers_control.py", copy=True)
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # faster downloads
            "HF_HUB_CACHE": CACHE_DIR,
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1",
            "MODAL_BATCH_SIZE": os.environ.get("MODAL_BATCH_SIZE", "10"),
            "MODAL_WAIT_MS": os.environ.get("MODAL_WAIT_MS", "500"),
            "MODAL_GPU": os.environ.get("MODAL_GPU", "A10G"),
            "MODAL_TIMEOUT_MINUTES": os.environ.get("MODAL_TIMEOUT_MINUTES", "3"),
        }
    )
)

# Import after defining image to ensure files are available
from sdxl_diffusers_control import setup_environment, load_models
from src.config import Config
from src.generation import process_single_generation


def download_and_save_image(url: str) -> str:
    """Download image from URL and save to a temporary file."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Create a temporary file with .png extension
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, "input.png")

        with open(temp_path, "wb") as f:
            f.write(response.content)

        return temp_path
    except Exception as e:
        raise ValueError(f"Failed to download image from URL: {e}")


cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("results", create_if_missing=True)


@app.cls(
    image=image,
    gpu=modal_settings.gpu,
    timeout=modal_settings.timeout,
    volumes={CACHE_DIR: cache_volume, RESULTS_DIR: results_volume},
    secrets=[
        modal.Secret.from_name("huggingface-token"),
        modal.Secret.from_name("visibl-secret"),
    ],
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    retries=1,
    max_containers=modal_settings.max_containers,
)
class ControlnetsInference:
    @modal.enter(snap=True)
    def load(self):
        logger.info("Loading base models (with snapshot)")
        # Initialize base configuration
        self.base_config = Config()
        
        # Setup environment and logging
        self.logger = setup_environment(self.base_config)
        
        # Load all models and store as instance variables
        self.pipeline, self.refiner, self.depth_estimator, self.feature_extractor = load_models(self.base_config, self.logger)

    def _upload_to_url(self, file_path: str, url: str):
        logger.info(f"Uploading to {url}")
        start_time = time.time()
        with open(file_path, "rb") as f:
            content_type = guess_image_content_type(url) or "image/webp"
            response = requests.put(
                url, data=f.read(), headers={"Content-Type": content_type}
            )
            response.raise_for_status()
        upload_time = time.time() - start_time
        logger.info(f"Upload completed in {upload_time:.2f} seconds")

        # Extract the base URL by removing query parameters
        parsed = urlparse(url)
        # Reconstruct URL without query parameters
        base_url = urlunparse(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                "",  # params
                "",  # query
                "",  # fragment
            )
        )
        return base_url

    def run(
        self,
        config_dict: dict = None,
        output_url: str = None,
        callback_url: str = None,
        result_key: str = None,
    ):
        # Create a config with overrides from config_dict
        if config_dict:
            # Handle URL input images
            if 'input_image' in config_dict and config_dict['input_image']:
                if config_dict['input_image'].startswith(("http://", "https://")):
                    config_dict['input_image'] = download_and_save_image(config_dict['input_image'])
            
            item_config = Config(config_dict)
        else:
            item_config = self.base_config
        
        # Process single generation using the loaded models
        try:
            success = process_single_generation(
                self.pipeline, 
                self.refiner, 
                self.depth_estimator, 
                self.feature_extractor, 
                item_config, 
                self.logger
            )
            
            if not success:
                raise Exception("Generation failed")
            
            # Get the output path from config (use refined output)
            result_path = item_config.refined_output

            # Process the results
            result = {}
            result_key = result_key or "0"
            
            # If output URL (typically a signed URL) provided, upload and return the URL
            # Otherwise just return the local path
            if output_url:
                self.logger.info("Using provided output URL for upload")
                result[result_key] = self._upload_to_url(result_path, output_url)
            else:
                self.logger.info(
                    f"No output URL provided, returning local path: {result_path}"
                )
                result[result_key] = result_path

            if callback_url:
                self._post_to_callback(
                    callback_url, {"status": "completed", "results": [result]}
                )

            return result
        except Exception as e:
            self.logger.error(f"Error in run method: {str(e)}")
            if callback_url:
                self._post_to_callback(
                    callback_url, {"status": "error", "error": str(e)}
                )
            raise

    def _post_to_callback(self, callback_url: str, data: dict):
        """Helper method to post data to callback URL"""
        try:
            # Convert any PosixPath objects to strings in the data
            def convert_paths(obj):
                if isinstance(obj, dict):
                    return {k: convert_paths(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_paths(item) for item in obj]
                elif hasattr(
                    obj, "__str__"
                ):  # This will catch PosixPath and other path-like objects
                    return str(obj)
                return obj

            serializable_data = convert_paths(data)

            # Get API token from environment
            callback_token = os.environ.get("CALLBACK_API_TOKEN")

            # Prepare headers with token if available
            headers = {"Content-Type": "application/json"}
            if callback_token:
                headers["Authorization"] = f"Bearer {callback_token}"

            start_time = time.time()
            response = requests.post(
                callback_url, json=serializable_data, headers=headers
            )
            response.raise_for_status()
            callback_time = time.time() - start_time
            logger.info(
                f"Successfully posted to callback URL: {callback_url} in {callback_time:.2f} seconds"
            )
        except Exception as e:
            logger.error(f"Failed to post to callback URL: {str(e)}", exc_info=True)

    @modal.batched(
        max_batch_size=modal_settings.max_batch_size, wait_ms=modal_settings.wait_ms
    )
    async def run_batch(self, input: list[dict]) -> list[str]:
        """Process a batch of inference requests"""
        # Use first callback URL for final callback with all results
        callback_url = input[0].get("callback_url") if input else None
        try:
            # Sort inputs by timestamp to maintain original request order
            sorted_inputs = sorted(input, key=lambda x: x.get("timestamp", 0))
            print(f"Sorted inputs: {[x.get('timestamp') for x in sorted_inputs]}")

            # Get the valid parameter names from the run method
            valid_params = {
                "config_dict",
                "output_url",
                "callback_url",
                "result_key",
            }
            
            # Process each input, extracting config and control parameters
            results = []
            for input_dict in sorted_inputs:
                # Separate config parameters from control parameters
                control_params = {}
                config_params = {}
                
                for k, v in input_dict.items():
                    if k in valid_params:
                        control_params[k] = v
                    else:
                        # Everything else goes into config
                        config_params[k] = v
                
                # Add config_params as config_dict if there are any
                if config_params:
                    control_params["config_dict"] = config_params
                
                results.append(self.run(**control_params))

            # If callback URL is provided, post results
            DISABLED_BATCH_CALLBACK = True
            if callback_url and not DISABLED_BATCH_CALLBACK:
                self._post_to_callback(
                    callback_url, {"status": "completed", "results": results}
                )

            # Modal batched function expects a list
            return results
        except Exception as e:
            logger.error(f"Error in batch inference: {str(e)}", exc_info=True)
            # If callback URL is provided, post error
            if callback_url:
                self._post_to_callback(
                    callback_url, {"status": "error", "error": str(e)}
                )
            raise e


# Should guess content type from a signed URL
def guess_image_content_type(file_path: str) -> str | None:
    # Initialize common image types explicitly
    image_mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
        ".tiff": "image/tiff",
        ".svg": "image/svg+xml",
        ".ico": "image/vnd.microsoft.icon",
    }

    # Handle URLs by extracting the path component
    if file_path.startswith(("http://", "https://")):
        parsed = urlparse(file_path)
        file_path = parsed.path

    # Get the file extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    # Return the matched MIME type or None if not an image
    return image_mime_types.get(ext)
