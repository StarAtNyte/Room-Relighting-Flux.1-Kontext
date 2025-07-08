import torch
import os
import json
from pathlib import Path
from diffusers import FluxKontextPipeline, FluxTransformer2DModel
from torchao.quantization import quantize_, int8_weight_only
from PIL import Image
import argparse
from tqdm import tqdm
import logging

class TimeBasedRoomProcessor:
    def __init__(self):
        self.pipe = None
        self.model_loaded = False
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('time_based_processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_model(self):
        """Load and initialize the FLUX.1-Kontext model with memory optimization"""
        try:
            self.logger.info("Loading FLUX.1-Kontext model...")
            
            # Load transformer with quantization for memory efficiency
            transformer = FluxTransformer2DModel.from_pretrained(
                "black-forest-labs/FLUX.1-Kontext-dev",
                subfolder="transformer",
                torch_dtype=torch.bfloat16
            )
            
            # Apply int8 quantization to reduce VRAM usage
            quantize_(transformer, int8_weight_only())
            
            # Load the pipeline
            self.pipe = FluxKontextPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Kontext-dev",
                transformer=transformer,
                torch_dtype=torch.bfloat16
            )
            
            # Enable CPU offloading to save VRAM
            self.pipe.enable_model_cpu_offload()
            
            self.model_loaded = True
            self.logger.info("Model loaded successfully!")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            self.logger.error("Please ensure you have:")
            self.logger.error("1. Latest diffusers: pip install -U git+https://github.com/huggingface/diffusers.git")
            self.logger.error("2. TorchAO: pip install torchao --extra-index-url https://download.pytorch.org/whl/cu121")
            self.logger.error("3. Access to the FLUX.1-Kontext-dev model on Hugging Face")
            self.model_loaded = False

    def get_time_based_lighting(self):
        """Define time-based lighting conditions with detailed shadow and lighting descriptions"""
        return {

            "morning_7am": {
                "prompt": "the same room at 7 AM morning with gentle warm sunlight streaming through windows at a low angle, medium-length shadows cast by furniture and objects across the floor, soft golden morning light illuminating surfaces, bright but not harsh lighting, shadows pointing away from window direction",
                "description": "Morning - 7 AM"
            },
            "late_morning_10am": {
                "prompt": "the same room at 10 AM late morning with bright sunlight coming through windows at a moderate angle, shorter shadows cast by objects, clear bright natural lighting, warm daylight filling the space, shadows becoming more compact and defined",
                "description": "Late Morning - 10 AM"
            },
            "midday_12pm": {
                "prompt": "the same room at 12 PM midday with intense bright sunlight streaming directly through windows, very short sharp shadows directly beneath objects, harsh bright daylight, strong contrast between lit and shadowed areas, minimal shadow length due to high sun position",
                "description": "Midday - 12 PM"
            },
            "afternoon_3pm": {
                "prompt": "the same room at 3 PM afternoon with bright sunlight through the windows,shadows cast in opposite direction from morning, afternoon light, clear visibility of all objects with well-defined shadows",
                "description": "Afternoon - 3 PM"
            },

            "Evening_7_pm": {
                "prompt": "Warm golden hour lighting filtering through windows, soft ambient evening glow, cozy twilight atmosphere, warm orange and yellow light tones, gentle shadows, comfortable evening ambiance, natural sunset lighting indoors",
                "description": "Evening - 7 PM "
            },
            "night_10pm": {
                "prompt": "Evening home lighting with light sources on, ceiling lights and table lamps providing comfortable ambient lighting, warm cozy interior illumination, well-lit living space for evening activities, soft overhead lighting filling the room, warm white light (3000K), functional evening lighting, clearly visible floor and furniture details, inviting home atmosphere",
                "description": "Night - 10 PM"
            },
            "cloudy_day": {
                "prompt": "the same room during a cloudy day with soft diffused natural light coming through windows, no direct sunlight, minimal soft shadows, even grey daylight illumination from overcast sky, objects clearly visible but with muted colors and soft lighting, no harsh shadows due to cloud cover",
                "description": "Cloudy Day"
            }
        }

    def get_supported_image_extensions(self):
        """Return supported image file extensions"""
        return {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

    def find_images_in_folder(self, folder_path):
        """Find all image files in a folder"""
        folder = Path(folder_path)
        if not folder.exists():
            return []
        
        supported_extensions = self.get_supported_image_extensions()
        images = []
        
        for file_path in folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                images.append(file_path)
        
        return sorted(images)

    def process_single_image(self, image_path, output_folder, times_to_generate, guidance_scale=2.5, num_inference_steps=25):
        """Process a single image with specified time-based lighting conditions"""
        if not self.model_loaded:
            self.logger.error("Model not loaded. Cannot process images.")
            return False

        try:
            # Load the input image
            input_image = Image.open(image_path)
            w, h = input_image.size
            
            time_conditions = self.get_time_based_lighting()
            
            for time_name in times_to_generate:
                if time_name not in time_conditions:
                    self.logger.warning(f"Unknown time condition: {time_name}")
                    continue
                
                output_path = output_folder / f"{time_name}.jpg"
                
                # Skip if output already exists
                if output_path.exists():
                    self.logger.info(f"Skipping {output_path} - already exists")
                    continue
                
                condition = time_conditions[time_name]
                prompt = condition["prompt"]
                self.logger.info(f"Generating {condition['description']} for {image_path.name}")
                
                # Generate the relit image
                result = self.pipe(
                    image=input_image,
                    prompt=prompt,
                    guidance_scale=guidance_scale,
                    height=h,
                    width=w,
                    num_inference_steps=num_inference_steps
                )
                
                output_image = result.images[0]
                output_image.save(output_path, quality=95)
                self.logger.info(f"Saved: {output_path}")
                
                # Clear CUDA cache to prevent memory issues
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {str(e)}")
            return False

    def process_input_folder(self, input_folder, output_folder, times_to_generate, guidance_scale=2.5, num_inference_steps=20):
        """Process Input folder (subfolders with single room images)"""
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        
        if not input_path.exists():
            self.logger.error(f"Input folder does not exist: {input_folder}")
            return
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Process subfolders containing single images
        subfolders = [f for f in input_path.iterdir() if f.is_dir()]
        
        if not subfolders:
            self.logger.error(f"No subfolders found in Input folder: {input_folder}")
            return
        
        for subfolder in tqdm(subfolders, desc="Processing Input subfolders"):
            self.logger.info(f"Processing Input subfolder: {subfolder.name}")
            
            # Find images in subfolder (should be single images)
            images = self.find_images_in_folder(subfolder)
            
            if not images:
                self.logger.warning(f"No images found in {subfolder}")
                continue
            
            # Create output subfolder
            output_subfolder = output_path / subfolder.name
            output_subfolder.mkdir(parents=True, exist_ok=True)
            
            # Process each image in the subfolder
            for image_path in images:
                # Create individual image output folder
                image_output_folder = output_subfolder / image_path.stem
                image_output_folder.mkdir(parents=True, exist_ok=True)
                
                self.logger.info(f"Processing image: {image_path}")
                success = self.process_single_image(
                    image_path, 
                    image_output_folder, 
                    times_to_generate,
                    guidance_scale,
                    num_inference_steps
                )
                
                if not success:
                    self.logger.error(f"Failed to process {image_path}")

    def process_room_images_folder(self, room_images_folder, output_folder, times_to_generate, guidance_scale=2.5, num_inference_steps=20):
        """Process Room Images folder (subfolders with day/night/cloudy versions of same room)"""
        room_path = Path(room_images_folder)
        output_path = Path(output_folder)
        
        if not room_path.exists():
            self.logger.error(f"Room Images folder does not exist: {room_images_folder}")
            return
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Process subfolders containing multiple versions (day/night/cloudy)
        subfolders = [f for f in room_path.iterdir() if f.is_dir()]
        
        if not subfolders:
            self.logger.error(f"No subfolders found in Room Images folder: {room_images_folder}")
            return
        
        for subfolder in tqdm(subfolders, desc="Processing Room Images subfolders"):
            self.logger.info(f"Processing Room Images subfolder: {subfolder.name}")
            
            # Find images in subfolder (should be day/night/cloudy versions)
            images = self.find_images_in_folder(subfolder)
            
            if not images:
                self.logger.warning(f"No images found in {subfolder}")
                continue
            
            self.logger.info(f"Found {len(images)} images in {subfolder.name}: {[img.name for img in images]}")
            
            # Create output subfolder
            output_subfolder = output_path / subfolder.name
            output_subfolder.mkdir(parents=True, exist_ok=True)
            
            # Process each base image (day/night/cloudy versions)
            for image_path in images:
                # Create a sub-subfolder for each base image variant
                base_name = image_path.stem
                image_output_folder = output_subfolder / base_name
                image_output_folder.mkdir(parents=True, exist_ok=True)
                
                self.logger.info(f"Processing base image: {image_path.name} -> {base_name}")
                success = self.process_single_image(
                    image_path,
                    image_output_folder,
                    times_to_generate,
                    guidance_scale,
                    num_inference_steps
                )
                
                if not success:
                    self.logger.error(f"Failed to process {image_path}")

    def save_processing_config(self, output_folder, config):
        """Save processing configuration for reference"""
        config_path = Path(output_folder) / "processing_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        self.logger.info(f"Saved processing config: {config_path}")

    def print_time_conditions(self):
        """Print all available time conditions with descriptions"""
        time_conditions = self.get_time_based_lighting()
        print("Available time-based lighting conditions:")
        print("=" * 50)
        for key, condition in time_conditions.items():
            print(f"  {key:15} - {condition['description']}")

def main():
    parser = argparse.ArgumentParser(description="Generate room images at different times of day with realistic lighting and shadows")
    
    parser.add_argument("--input_folder", type=str, 
                       help="Path to Input folder (subfolders with single room images)", default = "/home/alternative/Projects/flux-kontext/Input")
    parser.add_argument("--room_images_folder", type=str, 
                       help="Path to Room Images folder (subfolders with day/night/cloudy versions)", default="/home/alternative/Projects/flux-kontext/Room Images")
    parser.add_argument("--output_folder", type=str, required=True, 
                       help="Path to output folder")
    parser.add_argument("--times", nargs='+', default=None, 
                       help="Specific times to generate (space-separated). Use --list to see options")
    parser.add_argument("--guidance_scale", type=float, default=2.5, 
                       help="Guidance scale for generation (default: 3)")
    parser.add_argument("--num_inference_steps", type=int, default=25, 
                       help="Number of inference steps (default: 25)")
    parser.add_argument("--list", action='store_true', 
                       help="List all available time conditions and exit")
    
    args = parser.parse_args()
    
    processor = TimeBasedRoomProcessor()
    
    # List available conditions if requested
    if args.list:
        processor.print_time_conditions()
        return
    
    # Check that at least one input folder is provided
    if not args.input_folder and not args.room_images_folder:
        print("Error: You must specify either --input_folder or --room_images_folder (or both)")
        print("Use --help for more information")
        return
    
    # Load the model
    processor.load_model()
    if not processor.model_loaded:
        print("Failed to load model. Exiting.")
        return
    
    # Determine which times to generate
    all_times = list(processor.get_time_based_lighting().keys())
    if args.times:
        times_to_generate = [t for t in args.times if t in all_times]
        invalid_times = [t for t in args.times if t not in all_times]
        if invalid_times:
            print(f"Warning: Invalid time conditions ignored: {invalid_times}")
            print("Use --list to see available options")
    else:
        times_to_generate = all_times
    
    if not times_to_generate:
        print("No valid time conditions to generate. Use --list to see available options.")
        return
    
    print(f"Will generate {len(times_to_generate)} time conditions:")
    time_conditions = processor.get_time_based_lighting()
    for time_name in times_to_generate:
        print(f"  - {time_name}: {time_conditions[time_name]['description']}")
    
    # Save processing configuration
    config = {
        "input_folder": args.input_folder,
        "room_images_folder": args.room_images_folder,
        "output_folder": args.output_folder,
        "times_generated": times_to_generate,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "time_descriptions": {k: v['description'] for k, v in time_conditions.items() if k in times_to_generate}
    }
    processor.save_processing_config(args.output_folder, config)
    
    # Process Input folder if provided
    if args.input_folder:
        print(f"\nProcessing Input folder: {args.input_folder}")
        print("(Structure: subfolders with single room images)")
        processor.process_input_folder(
            args.input_folder,
            Path(args.output_folder) / "input_processed",
            times_to_generate,
            args.guidance_scale,
            args.num_inference_steps
        )
    
    # Process Room Images folder if provided
    if args.room_images_folder:
        print(f"\nProcessing Room Images folder: {args.room_images_folder}")
        print("(Structure: subfolders with day/night/cloudy versions)")
        processor.process_room_images_folder(
            args.room_images_folder,
            Path(args.output_folder) / "room_images_processed",
            times_to_generate,
            args.guidance_scale,
            args.num_inference_steps
        )
    
    print("\nProcessing completed!")
    print(f"Results saved in: {args.output_folder}")

if __name__ == "__main__":
    main()

# Example usage:
# Process only Input folder (single images per subfolder)
# python time_based_room_lighting.py --input_folder "./Input" --output_folder "./time_variations"

# Process only Room Images folder (day/night/cloudy versions per subfolder)  
# python time_based_room_lighting.py --room_images_folder "./Room Images" --output_folder "./time_variations"

# Process both folders
# python time_based_room_lighting.py --input_folder "./Input" --room_images_folder "./Room Images" --output_folder "./time_variations"

# Process specific times only
# python time_based_room_lighting.py --input_folder "./Input" --output_folder "./time_variations" --times dawn_5am midday_12pm evening_7pm night_10pm

# List all available time conditions
# python time_based_room_lighting.py --list