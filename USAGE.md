# 📖 Usage Guide

This guide provides detailed instructions for using Room Relighting AI in different scenarios.

## 🎯 Quick Start Options

### 1. Web Interface (Easiest)
Perfect for single images and experimentation.

```bash
python app.py
```

Navigate to `http://localhost:7600` and:
1. Upload a room image
2. Select a lighting condition
3. Adjust settings if needed
4. Click "Generate Relit Image"

### 2. Command Line (Batch Processing)
Ideal for processing multiple rooms locally.

```bash
# Basic usage
python relit_image.py --input_folder "./Input" --output_folder "./results"

# With custom settings
python relit_image.py \
  --input_folder "./Input" \
  --room_images_folder "./Room Images" \
  --output_folder "./results" \
  --guidance_scale 4.0 \
  --num_inference_steps 40 \
  --times dawn_5am midday_12pm evening_7pm night_10pm
```

### 3. Cloud Processing (Scalable)
Best for large batches and when you need more compute power.

```bash
# Upload data to Modal
modal volume put room-input-images ./Input /input/Input
modal volume put room-input-images "./Room Images" "/input/Room Images"

# Process in the cloud
modal run hourly_lighting.py --input-folder /input/Input

# Download results
modal volume get room-output-images / ./results
```

## 📁 Input Folder Structure

### Option A: Single Images per Room
```
Input/
├── Living_Room_A/
│   └── room.jpg
├── Bedroom_B/
│   └── bedroom.jpg
└── Kitchen_C/
    └── kitchen.jpg
```

### Option B: Multiple Versions per Room
```
Room Images/
├── Living_Room_A/
│   ├── day.jpg
│   ├── night.jpg
│   └── cloudy.jpg
├── Bedroom_B/
│   ├── morning.jpg
│   └── evening.jpg
```

## ⚙️ Configuration Options

### Processing Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `guidance_scale` | 2.5 | 1.0-10.0 | How closely to follow the prompt |
| `num_inference_steps` | 20 | 10-100 | Quality vs speed trade-off |
| `times_to_generate` | All 17 | Custom list | Specific lighting conditions |

### Guidance Scale Guidelines
- **1.0-2.0**: More creative, less adherence to prompt
- **2.5-3.5**: Balanced (recommended)
- **4.0-6.0**: Strong adherence to prompt
- **7.0+**: Very strict, may reduce quality

### Inference Steps Guidelines
- **10-20**: Fast, good for testing
- **20-30**: Balanced quality/speed (recommended)
- **40-50**: High quality, slower
- **50+**: Diminishing returns

## 🕐 Available Time Conditions

### Hourly Progression (17 conditions)
```python
times = [
    "dawn_5am", "morning_6am", "morning_7am", "morning_8am",
    "morning_9am", "late_morning_10am", "late_morning_11am",
    "midday_12pm", "afternoon_1pm", "afternoon_2pm", "afternoon_3pm",
    "late_afternoon_4pm", "late_afternoon_5pm", "evening_6pm",
    "evening_7pm", "night_8pm", "night_9pm", "night_10pm"
]
```

### Special Conditions
```python
special_conditions = [
    "golden_hour", "night_interior", "bright_evening",
    "moonlit_night", "cloudy_day", "rainy_day",
    "sunset", "dawn", "candlelit", "modern_led",
    "vintage_warm", "dramatic_spotlights"
]
```

## 📊 Output Structure

```
results/
├── input_processed/           # From Input folder
│   └── Living_Room_A/
│       ├── 0600.jpg          # 6 AM
│       ├── 0700.jpg          # 7 AM
│       ├── ...
│       ├── 2200.jpg          # 10 PM
│       └── timelapse.gif     # Animated sequence
└── room_images_processed/     # From Room Images folder
    └── [similar structure]
```

## 🎨 Advanced Usage

### Custom Lighting Prompts
```python
# Using the web interface
custom_prompt = "with warm candlelight and fairy lights"

# Using command line (modify relit_image.py)
custom_conditions = {
    "cyberpunk": "with neon lighting and cyberpunk atmosphere",
    "vintage": "with warm vintage Edison bulbs",
    "studio": "with professional studio lighting setup"
}
```

### Batch Processing with Filtering
```bash
# Process only specific times
python relit_image.py \
  --input_folder "./Input" \
  --times dawn_5am midday_12pm golden_hour night_interior

# Process with high quality settings
python relit_image.py \
  --input_folder "./Input" \
  --guidance_scale 4.5 \
  --num_inference_steps 50
```

### Resume Interrupted Processing
The system automatically skips already processed images:

```bash
# If processing was interrupted, just run the same command again
python relit_image.py --input_folder "./Input" --output_folder "./results"
# Will skip existing files and continue from where it left off
```

## 🔧 Troubleshooting

### Common Issues

#### Out of Memory (CUDA OOM)
```bash
# Reduce batch size or use CPU
export CUDA_VISIBLE_DEVICES=""  # Force CPU usage
# Or reduce inference steps
python relit_image.py --num_inference_steps 15
```

#### Model Download Issues
```bash
# Pre-download the model
python -c "from diffusers import FluxKontextPipeline; FluxKontextPipeline.from_pretrained('black-forest-labs/FLUX.1-Kontext-dev')"
```

#### Poor Quality Results
- Increase `guidance_scale` to 3.5-4.5
- Increase `num_inference_steps` to 30-40
- Ensure input images are high quality
- Try different lighting conditions

#### Slow Processing
- Reduce `num_inference_steps` to 15-20
- Use fewer time conditions
- Consider cloud processing for large batches

### Performance Optimization

#### For Better Speed
```python
# Use fewer inference steps
--num_inference_steps 15

# Process only key times
--times dawn_5am midday_12pm evening_7pm night_10pm

# Use lower guidance scale
--guidance_scale 2.0
```

#### For Better Quality
```python
# Use more inference steps
--num_inference_steps 40

# Higher guidance scale
--guidance_scale 4.0

# Ensure high-quality input images (1024px+ recommended)
```

## 📈 Monitoring Progress

### Local Processing
- Progress bars show current image and time condition
- Logs are saved to `time_based_processing.log`
- Check output folder for intermediate results

### Cloud Processing
```bash
# Monitor Modal job
modal app logs hourly-room-lighting-timelapse

# Check volume contents
modal volume ls room-output-images --recursive
```

## 🎯 Best Practices

### Input Image Guidelines
- **Resolution**: 1024px or higher for best results
- **Format**: JPG or PNG
- **Content**: Clear room interiors with visible windows
- **Lighting**: Any lighting condition works as input
- **Composition**: Avoid extreme close-ups or wide angles

### Processing Workflow
1. **Test first**: Use web interface to test single images
2. **Optimize settings**: Find best guidance_scale for your images
3. **Batch process**: Use command line for multiple images
4. **Review results**: Check quality before large batches
5. **Scale up**: Use cloud processing for production

### Output Management
- Results can be large (17 images + GIF per room)
- Consider processing subsets for testing
- Use cloud storage for large result sets
- Organize outputs by project or date

---

Need more help? Check our [GitHub Issues](https://github.com/yourusername/room-relighting-ai/issues) or [Discussions](https://github.com/yourusername/room-relighting-ai/discussions)!