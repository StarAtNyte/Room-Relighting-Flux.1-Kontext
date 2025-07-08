import modal
import torch
import json
from pathlib import Path
from diffusers import FluxKontextPipeline
from PIL import Image, ImageDraw, ImageFont
import logging
from typing import List, Dict, Any, Tuple
import os
import math

# Define the Modal app
app = modal.App("hourly-room-lighting-timelapse")

# Create the Modal image with required dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        [
            "torch",
            "torchvision",
            "diffusers>=0.30.0",
            "transformers",
            "accelerate",
            "pillow",
            "tqdm",
            "sentencepiece"
        ]
    )
    # The dev version of diffusers is needed for FLUX models
    .pip_install("git+https://github.com/huggingface/diffusers.git")
)

input_volume = modal.Volume.from_name("room-input-images", create_if_missing=True)
output_volume = modal.Volume.from_name("room-output-images", create_if_missing=True)


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


def calculate_sun_position(hour: int) -> Tuple[str, str, str]:
    """Calculate sun position and lighting characteristics for given hour"""
    # Sun angle calculation (simplified model)
    # Sunrise around 6 AM, sunset around 6 PM, peak at noon
    if hour < 6:
        return "pre-dawn", "very low", "cool"
    elif hour == 6:
        return "very low", "long", "warm golden"
    elif hour <= 8:
        return "low", "long", "warm golden"
    elif hour <= 10:
        return "medium-low", "medium-long", "warm"
    elif hour == 12:
        return "high overhead", "very short", "neutral bright"
    elif hour <= 14:
        return "high", "short", "neutral bright"
    elif hour <= 16:
        return "medium-high", "medium", "warm"
    elif hour <= 18:
        return "medium", "medium-long", "warm golden"
    elif hour <= 19:
        return "low", "long", "warm golden orange"
    elif hour <= 20:
        return "very low", "very long", "deep orange"
    else:  # Night hours
        return "below horizon", "none", "artificial interior"


def get_hourly_lighting_conditions() -> Dict[str, Dict[str, str]]:
    """Define physically accurate hourly lighting conditions from 6 AM to 10 PM"""
    conditions = {}
    
    for hour in range(6, 23):  # 6 AM to 10 PM
        sun_angle, shadow_length, color_temp = calculate_sun_position(hour)
        
        if hour <= 6:
            # Early morning
            prompt = f"Change only lighting to {hour}:00 AM early morning: very soft warm golden sunlight, long gentle shadows from {sun_angle} sun angle, preserve all furniture and room structure exactly, natural early morning ambiance"
            
        elif hour <= 8:
            # Morning
            prompt = f"Change only lighting to {hour}:00 AM morning: warm golden sunlight streaming naturally, {shadow_length} shadows cast from {sun_angle} sun position, preserve all furniture and room layout exactly, bright morning light"
            
        elif hour <= 11:
            # Late morning
            prompt = f"Change only lighting to {hour}:00 AM late morning: bright natural sunlight, {shadow_length} shadows from {sun_angle} sun, preserve all furniture and room structure exactly, clear bright daylight"
            
        elif hour == 12:
            # Noon - overhead sun
            prompt = f"Change only lighting to 12:00 PM noon: intense overhead sunlight, {shadow_length} sharp shadows directly below objects, preserve all furniture and room layout exactly, bright midday light"
            
        elif hour <= 14:
            # Early afternoon
            prompt = f"Change only lighting to {hour}:00 PM early afternoon: bright afternoon sunlight, {shadow_length} shadows from {sun_angle} sun position, preserve all furniture and room structure exactly, clear afternoon light"
            
        elif hour <= 16:
            # Afternoon
            prompt = f"Change only lighting to {hour}:00 PM afternoon: warm afternoon sunlight, {shadow_length} shadows cast from {sun_angle} sun angle, preserve all furniture and room layout exactly, pleasant afternoon lighting"
            
        elif hour <= 18:
            # Late afternoon
            prompt = f"Change only lighting to {hour}:00 PM late afternoon: warm golden afternoon light, {shadow_length} shadows from {sun_angle} sun position, preserve all furniture and room structure exactly, golden hour beginning"
            
        elif hour <= 19:
            # Evening golden hour
            prompt = f"Change only lighting to {hour}:00 PM evening: warm golden sunset light filtering through windows, {shadow_length} dramatic shadows from {sun_angle} sun, preserve all furniture and room layout exactly, beautiful golden hour lighting"
            
        elif hour <= 20:
            # Late evening
            prompt = f"Change only lighting to {hour}:00 PM late evening: soft warm orange sunset glow, {shadow_length} shadows from very low sun, preserve all furniture and room structure exactly, sunset ambiance"
            
        else:
            # Night (21:00-22:00)
            prompt = f"Change only lighting to {hour}:00 PM night: warm interior artificial lighting with ceiling lights and lamps on, dark outside through windows, cozy evening indoor lighting, preserve all furniture and room layout exactly, comfortable night ambiance"
        
        # Format hour for filename
        time_key = f"{hour:02d}00"  # e.g., "0600", "1200", "2200"
        hour_12 = hour if hour <= 12 else hour - 12
        am_pm = "AM" if hour < 12 else "PM"
        if hour_12 == 0:
            hour_12 = 12
            
        conditions[time_key] = {
            "prompt": prompt,
            "description": f"{hour_12}:00 {am_pm}",
            "hour": hour,
            "sun_angle": sun_angle,
            "shadow_length": shadow_length,
            "color_temp": color_temp
        }
    
    return conditions


def get_supported_image_extensions():
    """Return supported image file extensions"""
    return {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def find_images_in_folder(folder_path: Path) -> List[Path]:
    """Find all image files in a folder"""
    if not folder_path.exists():
        return []

    supported_extensions = get_supported_image_extensions()
    images = []

    for file_path in folder_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            images.append(file_path)

    return sorted(images)


def is_processing_complete(output_folder: Path) -> bool:
    """Check if all hourly images and GIF have been generated"""
    if not output_folder.exists():
        return False
    
    time_conditions = get_hourly_lighting_conditions()
    
    # Check if all hourly images exist
    for time_key in time_conditions.keys():
        image_path = output_folder / f"{time_key}.jpg"
        if not image_path.exists():
            return False
    
    # Check if GIF exists
    gif_path = output_folder / "timelapse.gif"
    if not gif_path.exists():
        return False
    
    return True


def create_timelapse_gif(image_folder: Path, output_path: Path, duration: int = 500):
    """Create a time-lapse GIF from hourly images"""
    try:
        images = []
        time_conditions = get_hourly_lighting_conditions()
        
        # Load images in chronological order
        for time_key in sorted(time_conditions.keys()):
            image_path = image_folder / f"{time_key}.jpg"
            if image_path.exists():
                img = Image.open(image_path)
                
                # Add timestamp overlay
                draw = ImageDraw.Draw(img)
                timestamp = time_conditions[time_key]["description"]
                
                # Try to use a default font, fallback to PIL default if not available
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
                except:
                    font = ImageFont.load_default()
                
                # Add semi-transparent background for text
                text_bbox = draw.textbbox((0, 0), timestamp, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # Position text in top-right corner
                x = img.width - text_width - 20
                y = 20
                
                # Draw background rectangle
                draw.rectangle([x-10, y-5, x+text_width+10, y+text_height+5], 
                             fill=(0, 0, 0, 128))
                
                # Draw timestamp
                draw.text((x, y), timestamp, fill=(255, 255, 255), font=font)
                
                images.append(img)
        
        if images:
            # Save as GIF
            images[0].save(
                output_path,
                save_all=True,
                append_images=images[1:],
                duration=duration,
                loop=0,
                optimize=True
            )
            return True
        
        return False
        
    except Exception as e:
        logging.error(f"Error creating GIF: {str(e)}")
        return False


@app.cls(
    gpu="A100-80GB",
    image=image,
    volumes={"/input": input_volume, "/output": output_volume},
    timeout=86400,  # 24 hours timeout (24 * 60 * 60 seconds)
    min_containers=1,
    max_containers=1,
    scaledown_window=300,
    secrets=[modal.Secret.from_name("flux-kontext-hf-token")]
)
class HourlyRoomProcessor:
    @modal.enter()
    def load_model(self):
        """Load and initialize the FLUX.1-Kontext model when container starts"""
        self.logger = setup_logging()
        try:
            self.logger.info("Loading FLUX.1-Kontext model on Modal...")

            self.pipe = FluxKontextPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Kontext-dev",
                torch_dtype=torch.bfloat16,
            )
            self.pipe.to("cuda")

            self.model_loaded = True
            self.logger.info("Model loaded successfully on Modal!")

        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            self.model_loaded = False
            self.pipe = None

    def _process_single_image_internal(
        self,
        image_path: Path,
        output_folder: Path,
        guidance_scale: float = 3.5,  # Higher for better prompt adherence
        num_inference_steps: int = 40,  # More steps for better quality
    ) -> bool:
        """Internal method to process a single image for all hourly conditions"""
        if not hasattr(self, "model_loaded") or not self.model_loaded:
            self.logger.error("Model not loaded. Cannot process images.")
            return False

        # Check if processing is already complete
        if is_processing_complete(output_folder):
            self.logger.info(f"⏭️  SKIPPING {image_path.name} - Already processed (found all 17 images + GIF)")
            return True

        try:
            input_image = Image.open(image_path).convert("RGB")
            width, height = input_image.size
            time_conditions = get_hourly_lighting_conditions()

            output_folder.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"🎯 Processing {len(time_conditions)} hourly conditions for {image_path.name}")

            # Check which images are already processed
            existing_images = []
            missing_images = []
            
            for time_key, condition in time_conditions.items():
                output_path = output_folder / f"{time_key}.jpg"
                if output_path.exists():
                    existing_images.append(time_key)
                else:
                    missing_images.append((time_key, condition))
            
            if existing_images:
                self.logger.info(f"📁 Found {len(existing_images)} existing images, processing {len(missing_images)} missing ones")
            
            # Process only missing images
            for time_key, condition in missing_images:
                output_path = output_folder / f"{time_key}.jpg"

                prompt = condition["prompt"]
                
                # Strong negative prompt to prevent any structural changes
                negative_prompt = (
                    "changing furniture, moving objects, different room layout, "
                    "adding furniture, removing furniture, changing walls, "
                    "different architecture, changing room structure, "
                    "different perspective, changing camera angle, "
                    "different objects, changing colors of objects, "
                    "overly warm colors, overly blue colors, oversaturation, "
                    "unrealistic lighting, artificial colors"
                )
                
                self.logger.info(f"🌅 Generating {condition['description']} ({condition['hour']}:00) - {len(missing_images) - missing_images.index((time_key, condition))} remaining")

                # Generate with settings optimized for lighting changes only
                result = self.pipe(
                    image=input_image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    height=height,
                    width=width,
                )

                output_image = result.images[0]
                
                # Ensure output dimensions match input exactly
                if output_image.size != (width, height):
                    output_image = output_image.resize((width, height), Image.Resampling.LANCZOS)
                
                output_image.save(output_path, quality=95, optimize=True)
                self.logger.info(f"💾 Saved: {output_path}")

                # Clear CUDA cache after each image to prevent memory issues
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Create time-lapse GIF (only if not already exists)
            gif_path = output_folder / "timelapse.gif"
            if not gif_path.exists():
                self.logger.info("🎬 Creating time-lapse GIF...")
                
                if create_timelapse_gif(output_folder, gif_path, duration=400):
                    self.logger.info(f"✅ Time-lapse GIF saved: {gif_path}")
                else:
                    self.logger.error("❌ Failed to create time-lapse GIF")
            else:
                self.logger.info("⏭️  Time-lapse GIF already exists")

            return True

        except Exception as e:
            self.logger.error(f"❌ Error processing {image_path}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    @modal.method()
    def process_single_image(
        self,
        image_path: str,
        output_folder: str,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 40,
    ) -> bool:
        """Process a single image with all hourly lighting conditions"""
        return self._process_single_image_internal(
            Path(image_path),
            Path(output_folder),
            guidance_scale,
            num_inference_steps,
        )

    @modal.method()
    def process_input_folder(
        self,
        input_folder: str,
        output_folder: str,
        guidance_scale: float = 4.0,
        num_inference_steps: int = 40,
    ) -> Dict[str, Any]:
        """Process Input folder (subfolders with single room images)"""
        input_path = Path(input_folder)
        output_path = Path(output_folder)

        if not input_path.exists():
            msg = f"Input folder does not exist: {input_folder}"
            self.logger.error(msg)
            return {"success": False, "error": msg}

        output_path.mkdir(parents=True, exist_ok=True)
        subfolders = [f for f in input_path.iterdir() if f.is_dir()]

        if not subfolders:
            msg = f"No subfolders found in Input folder: {input_folder}"
            self.logger.error(msg)
            return {"success": False, "error": msg}

        processed_count = 0
        failed_count = 0
        skipped_count = 0

        for subfolder in subfolders:
            self.logger.info(f"📂 Processing Input subfolder: {subfolder.name}")
            images = find_images_in_folder(subfolder)

            if not images:
                self.logger.warning(f"⚠️  No images found in {subfolder}")
                continue
            
            output_subfolder = output_path / subfolder.name
            output_subfolder.mkdir(parents=True, exist_ok=True)

            for image_path in images:
                image_output_folder = output_subfolder / image_path.stem
                self.logger.info(f"🖼️  Processing image: {image_path}")
                
                # Check if already processed
                if is_processing_complete(image_output_folder):
                    self.logger.info(f"⏭️  SKIPPING {image_path.name} - Already processed")
                    skipped_count += 1
                    continue
                
                success = self._process_single_image_internal(
                    image_path,
                    image_output_folder,
                    guidance_scale,
                    num_inference_steps,
                )

                if success:
                    processed_count += 1
                else:
                    failed_count += 1
                    self.logger.error(f"❌ Failed to process {image_path}")

        return {
            "success": True,
            "processed_count": processed_count,
            "failed_count": failed_count,
            "skipped_count": skipped_count,
            "total_subfolders": len(subfolders),
        }

    @modal.method()
    def process_room_images_folder(
        self,
        room_images_folder: str,
        output_folder: str,
        guidance_scale: float = 4.0,
        num_inference_steps: int = 40,
    ) -> Dict[str, Any]:
        """Process Room Images folder (subfolders with multiple versions of same room)"""
        room_path = Path(room_images_folder)
        output_path = Path(output_folder)

        if not room_path.exists():
            msg = f"Room Images folder does not exist: {room_images_folder}"
            self.logger.error(msg)
            return {"success": False, "error": msg}

        output_path.mkdir(parents=True, exist_ok=True)
        subfolders = [f for f in room_path.iterdir() if f.is_dir()]

        if not subfolders:
            msg = f"No subfolders found in Room Images folder: {room_images_folder}"
            self.logger.error(msg)
            return {"success": False, "error": msg}

        processed_count = 0
        failed_count = 0
        skipped_count = 0
        
        for subfolder in subfolders:
            self.logger.info(f"📂 Processing Room Images subfolder: {subfolder.name}")
            
            # Find and sort images in the subfolder
            images = find_images_in_folder(subfolder)
            
            if len(images) < 2:
                self.logger.warning(f"⚠️  Skipping {subfolder.name}: found {len(images)} images, but need at least 2 to select the second one.")
                continue
            
            # Select the second image from the sorted list (index 1)
            image_to_process = images[1]
            
            self.logger.info(f"🔍 Found {len(images)} images. Selecting the second one to process: {image_to_process.name}")
            
            # Create output subfolder for this room
            output_subfolder = output_path / subfolder.name
            output_subfolder.mkdir(parents=True, exist_ok=True)
            
            # Check if already processed
            if is_processing_complete(output_subfolder):
                self.logger.info(f"⏭️  SKIPPING {subfolder.name} - Already processed")
                skipped_count += 1
                continue
            
            success = self._process_single_image_internal(
                image_to_process,
                output_subfolder,
                guidance_scale,
                num_inference_steps,
            )
            
            if success:
                processed_count += 1
            else:
                failed_count += 1
                self.logger.error(f"❌ Failed to process base image for subfolder {subfolder.name}")
        
        return {
            "success": True,
            "processed_count": processed_count,
            "failed_count": failed_count,
            "skipped_count": skipped_count,
            "total_subfolders": len(subfolders)
        }


# Utility functions
@app.function(image=image)
def list_hourly_conditions():
    """List all available hourly conditions"""
    time_conditions = get_hourly_lighting_conditions()

    print("Hourly lighting conditions (6 AM - 10 PM):")
    print("=" * 60)
    for key, condition in time_conditions.items():
        hour = condition['hour']
        sun_info = f"Sun: {condition['sun_angle']}, Shadows: {condition['shadow_length']}"
        print(f"  {condition['description']:10} ({hour:02d}:00) - {sun_info}")
    
    print(f"\nTotal conditions: {len(time_conditions)}")
    print("Each image will generate 17 hourly variations + 1 time-lapse GIF")


@app.function(
    image=image, 
    volumes={"/input": input_volume, "/output": output_volume}, 
    timeout=86400,  # 24 hours timeout (24 * 60 * 60 seconds)
    secrets=[modal.Secret.from_name("flux-kontext-hf-token")]
)
def process_images(
    input_folder: str = None,
    room_images_folder: str = None,
    output_folder: str = "/output",
    guidance_scale: float = 4.0,
    num_inference_steps: int = 40,
):
    """Main processing function for both Input and Room Images folders"""
    if not input_folder and not room_images_folder:
        return {
            "success": False,
            "error": "You must specify either --input-folder or --room-images-folder (or both)",
        }

    # Debug: Print volume contents
    print("Contents of /input:")
    print("=" * 50)
    try:
        import os
        for root, dirs, files in os.walk("/input"):
            level = root.replace("/input", "").count(os.sep)
            indent = " " * 2 * level
            print(f"{indent}📁 {os.path.basename(root)}/")
            subindent = " " * 2 * (level + 1)
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    size = os.path.getsize(file_path)
                    size_mb = size / (1024 * 1024)
                    print(f"{subindent}📄 {file} ({size_mb:.2f} MB)")
                except:
                    print(f"{subindent}📄 {file}")
    except Exception as e:
        print(f"Error listing contents: {e}")

    time_conditions = get_hourly_lighting_conditions()
    print(f"\n🎯 Will generate {len(time_conditions)} hourly lighting conditions + time-lapse GIFs:")
    
    for time_key, condition in time_conditions.items():
        print(f"  - {condition['description']} ({condition['hour']:02d}:00)")

    processor = HourlyRoomProcessor()
    
    results = {}
    if input_folder:
        print(f"\n📂 Processing Input folder: {input_folder}")
        input_output_folder = f"{output_folder}/input_processed"
        input_result = processor.process_input_folder.remote(
            input_folder,
            input_output_folder,
            guidance_scale,
            num_inference_steps,
        )
        results["input_folder"] = input_result

    if room_images_folder:
        print(f"\n🏠 Processing Room Images folder: {room_images_folder}")
        room_output_folder = f"{output_folder}/room_images_processed"
        room_result = processor.process_room_images_folder.remote(
            room_images_folder,
            room_output_folder,
            guidance_scale,
            num_inference_steps,
        )
        results["room_images_folder"] = room_result

    # Save processing configuration
    config = {
        "input_folder": input_folder,
        "room_images_folder": room_images_folder,
        "output_folder": output_folder,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "hourly_conditions": len(time_conditions),
        "time_range": "6 AM - 10 PM",
        "timeout_hours": 24,
        "time_descriptions": {
            k: v["description"]
            for k, v in time_conditions.items()
        },
        "results": results,
    }

    config_path = Path(output_folder) / "processing_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Commit volumes to persist changes
    print("\n💾 Committing output volume...")
    output_volume.commit()
    print("✅ Commit complete.")

    print("\n🎉 Processing completed!")
    print(f"Generated {len(time_conditions)} hourly images + time-lapse GIFs for each input")
    print(f"Configuration saved to: {config_path}")
    return {"success": True, "results": results}


@app.local_entrypoint()
def main(
    input_folder: str = None,
    room_images_folder: str = None,
    guidance_scale: float = 4.0,
    num_inference_steps: int = 40,
    list_conditions: bool = False,
):
    """
    Main CLI entrypoint for Modal - Hourly Time-lapse Room Lighting Generator

    \b
    Generates physically accurate lighting from 6 AM to 10 PM (every hour) plus time-lapse GIFs.
    
    \b
    Features:
    - 17 hourly lighting conditions with accurate sun positions and shadow angles
    - Preserves room structure and furniture placement exactly
    - Maintains original image dimensions
    - Generates time-lapse GIF with timestamps for each room
    - Natural color temperature progression throughout the day
    - Proper night lighting with interior illumination
    - 24-hour timeout for long processing jobs
    - Smart skip logic - resumes from where it left off

    \b
    Step-by-Step Usage:
    
    \b
    STEP 1: Upload your images to Modal volume
    
    # Create and upload to input volume
    # For a folder structure like: ./my_rooms/bedroom/image1.jpg, ./my_rooms/kitchen/image2.jpg
    modal volume put room-input-images ./my_rooms /input
    
    # Or upload specific folders:
    modal volume put room-input-images ./Input_Folder /input/Input
    modal volume put room-input-images "./Room Images Folder" "/input/Room Images"
    
    # Check what was uploaded:
    modal volume ls room-input-images
    modal volume ls room-input-images /input

    \b
    STEP 2: List available hourly conditions (optional)
    modal run hourly_lighting.py --list-conditions

    \b
    STEP 3: Process your uploaded images
    
    # Process Input folder (subfolders with individual room images)
    modal run hourly_lighting.py --input-folder /input/Input
    
    # Process Room Images folder (subfolders with multiple versions of same room)
    modal run hourly_lighting.py --room-images-folder "/input/Room Images"
    
    # Process both with custom settings for better structure preservation
    modal run hourly_lighting.py --input-folder /input/Input --room-images-folder "/input/Room Images" --guidance-scale 4.5
    
    # Process any folder structure you uploaded
    modal run hourly_lighting.py --input-folder /input

    \b
    STEP 4: Download processed results
    
    # Download everything
    modal volume get room-output-images / ./local_results_directory
    
    # Or download specific results:
    modal volume get room-output-images /input_processed ./input_results
    modal volume get room-output-images /room_images_processed ./room_results
    
    # Download only GIFs:
    modal volume get room-output-images --include="*.gif" / ./gifs_only
    
    # List what's available to download:
    modal volume ls room-output-images
    modal volume ls room-output-images --recursive

    \b
    Resume Processing:
    If processing times out or is interrupted, simply run the same command again.
    The script will automatically skip already processed images and resume from where it left off.

    \b
    Output Structure:
    - Each input image generates 17 hourly variations (0600.jpg to 2200.jpg)
    - One time-lapse GIF (timelapse.gif) with timestamps
    - All images maintain original dimensions exactly
    
    \b
    Parameters:
    --guidance-scale: Higher values (4.0-5.0) follow prompts more precisely
    --num-inference-steps: More steps (40-50) for better quality
    """

    if list_conditions:
        list_hourly_conditions.remote()
        return

    if input_folder or room_images_folder:
        process_images.remote(
            input_folder=input_folder,
            room_images_folder=room_images_folder,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )
        print("\n🚀 Hourly time-lapse generation job submitted to Modal.")
        print("⏱️  This has a 24-hour timeout and will automatically skip already processed images.")
        print("📊 This will generate 17 hourly images + GIF for each input image.")
        print("🔍 Run `modal app logs hourly-room-lighting-timelapse` to see live progress.")
        print("\n📥 TO DOWNLOAD RESULTS WHEN COMPLETE:")
        print("   modal volume get room-output-images / ./my_results")
        print("   modal volume get room-output-images /input_processed ./input_results")
        print("   modal volume get room-output-images /room_images_processed ./room_results")
        print("   modal volume get room-output-images --include='*.gif' / ./gifs_only")
        print("\n🔄 If processing is interrupted, run the same command again to resume.")
    else:
        print("❌ Error: Please specify --input-folder and/or --room-images-folder.")
        print("💡 Use --help for detailed usage information.")
        print("\n📥 QUICK DOWNLOAD COMMANDS:")
        print("   # List what's available:")
        print("   modal volume ls room-output-images")
        print("   modal volume ls room-output-images --recursive")
        print("   ")
        print("   # Download everything:")
        print("   modal volume get room-output-images / ./all_results")
        print("   ")
        print("   # Download specific folders:")
        print("   modal volume get room-output-images /input_processed ./input_results")
        print("   modal volume get room-output-images /room_images_processed ./room_results")
        print("   ")
        print("   # Download only GIFs:")
        print("   modal volume get room-output-images --include='*.gif' / ./gifs_only")
        print("   ")
        print("   # Download a specific room:")
        print("   modal volume get room-output-images /input_processed/bedroom ./bedroom_results")