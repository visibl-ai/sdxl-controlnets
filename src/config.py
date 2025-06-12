import logging
import json
import torch


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
        self.depth_model = "Intel/dpt-hybrid-midas"
        
        # Cache and environment
        self.cache_dir = "./cache"
        self.device = "cuda"
        self.torch_dtype = torch.bfloat16
        self.local_files_only = False
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