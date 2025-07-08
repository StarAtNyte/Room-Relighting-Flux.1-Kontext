# Room Relighting

Transform your room images with realistic lighting conditions using FLUX.1-Kontext AI model. 

![Room Relighting Demo](assets/demo-banner.png)

## ✨ Features

- **17 Hourly Lighting Conditions**: From 6 AM to 10 PM with realistic sun angles and shadows
- **Multiple Lighting Scenarios**: Golden hour, night interior, cloudy day, rainy weather, and more
- **Automatic Time-lapse Generation**: Creates smooth GIF animations showing lighting transitions
- **Batch Processing**: Process multiple rooms simultaneously
- **Memory Optimized**: Uses int8 quantization for efficient VRAM usage
- **Web Interface**: Easy-to-use Gradio interface for single image processing
- **Cloud Processing**: Modal.com integration for scalable batch processing

## 🎯 Use Cases

- **Interior Design**: Visualize how rooms look at different times of day
- **Real Estate**: Show properties in various lighting conditions
- **Architecture**: Study natural lighting patterns in spaces
- **Photography**: Plan optimal shooting times for interior photography
- **Art & Visualization**: Create stunning time-lapse sequences

## 🚀 Quick Start

### Option 1: Web Interface (Single Images)

```bash
# Install dependencies
pip install -U git+https://github.com/huggingface/diffusers.git
pip install torchao --extra-index-url https://download.pytorch.org/whl/cu121
pip install torch torchvision pillow gradio tqdm

# Run the web interface
python app.py
```

Open your browser to `http://localhost:7600` and start relighting rooms!

### Option 2: Batch Processing (Local)

```bash
# Process single room images
python relit_image.py --input_folder "./Input" --output_folder "./results"

# Process room collections (multiple versions per room)
python relit_image.py --room_images_folder "./Room Images" --output_folder "./results"

# Process both with custom settings
python relit_image.py --input_folder "./Input" --room_images_folder "./Room Images" --output_folder "./results" --guidance_scale 4.0 --num_inference_steps 40
```

### Option 3: Cloud Processing (Modal.com)

```bash
# Setup Modal volumes
modal volume put room-input-images ./Input /input/Input
modal volume put room-input-images "./Room Images" "/input/Room Images"

# Process in the cloud
modal run hourly_lighting.py --input-folder /input/Input --room-images-folder "/input/Room Images"

# Download results
modal volume get room-output-images / ./results
```

## 📁 Project Structure

```
room-relighting-ai/
├── app.py                    # Gradio web interface
├── relit_image.py           # Local batch processing
├── hourly_lighting.py       # Modal.com cloud processing
├── requirements.txt         # Python dependencies
├── examples/               # Sample input/output images
│   ├── input/             # Sample room images
│   └── output/            # Generated results
└── assets/                # Documentation assets
    └── demo-banner.png    # Project banner
```

## 🎨 Lighting Conditions

### Hourly Conditions (6 AM - 10 PM)
- **6-8 AM**: Soft morning light with long shadows
- **9-11 AM**: Bright morning sunlight
- **12 PM**: Intense midday lighting with short shadows
- **1-3 PM**: Bright afternoon light
- **4-6 PM**: Warm afternoon golden light
- **7-8 PM**: Golden hour with dramatic shadows
- **9-10 PM**: Evening interior lighting

### Special Conditions
- **Golden Hour**: Warm golden light streaming through windows
- **Night Interior**: Cozy lamps and overhead lights
- **Bright Evening**: All interior lights on, well-lit space
- **Moonlit Night**: Cool blue moonlight
- **Cloudy Day**: Soft diffused lighting
- **Rainy Day**: Grey light with rain effects
- **Custom**: Define your own lighting scenario

## 📊 Results Gallery

### Bathroom Lighting Study
![Bathroom Timelapse](examples/output/Bathroom_Innorim_timelapse.gif)

*Bathroom Innorim - Modern bathroom with natural lighting progression*

### Bedroom Lighting Variations  
![Bedroom Timelapse](examples/output/Bedroom_Arcadu_timelapse.gif)

*Bedroom Arcadu - Cozy bedroom showing 17-hour lighting cycle*

### Contemporary Bedroom
![Bedroom Timelapse](examples/output/Bedroom_Aspilig_S2_timelapse.gif)

*Bedroom Aspilig S2 - Contemporary bedroom with dramatic lighting changes*

## 🛠️ Technical Details

### Model Information
- **Base Model**: FLUX.1-Kontext-dev by Black Forest Labs
- **Optimization**: int8 quantization for memory efficiency
- **Input Resolution**: Automatically handled by the model
- **Processing Time**: ~30-60 seconds per image (depending on hardware)

### System Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended)
- **RAM**: 16GB+ system RAM
- **Storage**: 10GB+ free space for model and results
- **Python**: 3.8+ with CUDA support

### Performance Optimization
- Uses quantized transformer for reduced VRAM usage
- Automatic memory management and cleanup
- Batch processing for efficiency
- Resume capability for interrupted processing

## 📝 Usage Examples

### Basic Single Image Processing
```python
from relit_image import TimeBasedRoomProcessor

processor = TimeBasedRoomProcessor()
processor.load_model()

# Process single image with all hourly conditions
processor.process_single_image(
    "bedroom.jpg", 
    "output/bedroom/", 
    times_to_generate=["dawn_5am", "midday_12pm", "evening_7pm"]
)
```

### Custom Lighting Conditions
```python
# Define custom lighting
custom_conditions = {
    "cyberpunk": "the same room with neon lighting and cyberpunk atmosphere",
    "vintage": "the same room with warm vintage lighting and retro bulbs"
}

# Apply custom lighting
result = processor.generate_relit_image(
    input_image, 
    custom_conditions["cyberpunk"],
    guidance_scale=3.5
)
```

## 🔧 Configuration

### Processing Parameters
- **guidance_scale**: Controls adherence to prompt (2.0-5.0, default: 2.5)
- **num_inference_steps**: Quality vs speed trade-off (20-50, default: 20)
- **times_to_generate**: Specific time conditions to process

### Input Folder Structure
```
Input/
├── Living_Room_A/
│   └── room_image.jpg
├── Bedroom_B/
│   └── bedroom.jpg
└── Kitchen_C/
    └── kitchen.jpg
```

### Output Structure
```
output/
├── input_processed/
│   └── Living_Room_A/
│       ├── 0600.jpg
│       ├── 0700.jpg
│       ├── ...
│       ├── 2200.jpg
│       └── timelapse.gif
└── room_images_processed/
    └── [similar structure]
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/yourusername/room-relighting-ai.git
cd room-relighting-ai
pip install -r requirements.txt
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Black Forest Labs** for the FLUX.1-Kontext model
- **Hugging Face** for the Diffusers library
- **Modal.com** for cloud processing infrastructure
- **Gradio** for the web interface framework

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/room-relighting-ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/room-relighting-ai/discussions)
- **Documentation**: [Wiki](https://github.com/yourusername/room-relighting-ai/wiki)

---

**Made with ❤️ for the AI and interior design community**

*Transform your spaces, one light ray at a time* ✨