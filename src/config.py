import logging
import json
import torch
import os
from .image_preprocessing import SUPPORTED_DIMENSIONS


class Config:
    """Configuration for SD3.5 ControlNet pipeline"""
    def __init__(self, json_config=None):
        # Model paths
        self.model_repo = "stabilityai/stable-diffusion-3.5-large"
        ## need to convert to diffusers format. 
        ## https://github.com/huggingface/diffusers/blob/6c7fad7ec8b2417c92326804e1751658874fd43b/scripts/convert_sd3_controlnet_to_diffusers.py#L2
        ## python scripts/convert_sd3_controlnet_to_diffusers.py --checkpoint_path "../sd3.5/models/sd3.5_large_controlnet_depth.safetensors" --output_path ../sd3.5/models/sd3.5_large_controlnet_depth_diffusers
        self.depth_controlnet_path = "moeadham/stable-diffusion-3.5-large-controlnet-depth-diffusers"
        self.canny_controlnet_path = "moeadham/stable-diffusion-3.5-large-controlnet-canny-diffusers"
        self.blur_controlnet_path = "moeadham/stable-diffusion-3.5-large-controlnet-blur-diffusers"
        
        # Depth model configuration
        self.depth_model_type = os.environ.get("DEPTH_MODEL_TYPE", "depth_anything_v2")  # Options: "dpt" or "depth_anything_v2"
        self.depth_model = "Intel/dpt-hybrid-midas"  # Used when depth_model_type is "dpt"
        self.depth_anything_model = os.environ.get("DEPTH_ANYTHING_MODEL", "depth-anything/Depth-Anything-V2-Large-hf")  # Used when depth_model_type is "depth_anything_v2"
        
        # Cache and environment
        self.cache_dir = "./cache"
        self.device = "cuda"
        self.torch_dtype = torch.bfloat16
        env_val = os.environ.get("LOCAL_FILES_ONLY", "true").lower()
        self.local_files_only = env_val in ("true", "1", "True") # false by default
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
        # Aspect ratio configuration
        self.aspect_ratio = "auto"  # "auto", "square", "landscape", "portrait"
        # Height and width will be set dynamically based on aspect ratio
        self.height = None  # Will be set during preprocessing
        self.width = None   # Will be set during preprocessing
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
    
    def get_target_aspect_ratio(self):
        """
        Get the target aspect ratio to use for preprocessing.
        Returns None if auto-detection should be used.
        """
        # If aspect_ratio is explicitly set to a valid option (not "auto"),
        # use it regardless of auto_aspect_ratio setting
        if self.aspect_ratio and self.aspect_ratio != "auto":
            if self.aspect_ratio in ["square", "landscape", "portrait"]:
                return self.aspect_ratio
            else:
                logger = logging.getLogger(__name__)
                logger.warning(f"Invalid aspect_ratio '{self.aspect_ratio}', using auto-detection")
                return None
        # Otherwise, use auto-detection
        return None
    
    def validate_dimensions(self):
        """Validate that dimensions are set and match supported values"""
        if self.height is None or self.width is None:
            raise ValueError("Height and width must be set before generation")
        
        # Check if dimensions match any supported configuration
        current_dims = (self.width, self.height)
        valid = any(dims == current_dims for dims in SUPPORTED_DIMENSIONS.values())
        
        if not valid:
            raise ValueError(
                f"Dimensions {self.width}x{self.height} not supported. "
                f"Supported dimensions: {list(SUPPORTED_DIMENSIONS.values())}"
            )
    
    @classmethod
    def from_json_file(cls, json_path):
        """Create Config instance from JSON file"""
        with open(json_path, 'r') as f:
            json_config = json.load(f)
        return cls(json_config)