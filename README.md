# Room Relighting

Transform your room images with realistic lighting conditions using FLUX.1-Kontext AI model. 

## ✨ Features

- **17 Hourly Lighting Conditions**: From 6 AM to 10 PM with realistic sun angles and shadows
- **Multiple Lighting Scenarios**: Golden hour, night interior, cloudy day, rainy weather, and more
- **Automatic Time-lapse Generation**: Creates smooth GIF animations showing lighting transitions


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

![example2](https://github.com/user-attachments/assets/e9740d86-4aa3-4bc9-bb83-c43b5d4a77ab)

![Example 1](https://github.com/user-attachments/assets/f7161e05-5594-49fc-8dc0-530ae462ad6a)


