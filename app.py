import torch
import gradio as gr
from diffusers import FluxKontextPipeline, FluxTransformer2DModel
from diffusers.utils import load_image
from torchao.quantization import quantize_, int8_weight_only
from PIL import Image
import numpy as np
import os

class RoomRelightingApp:
    def __init__(self):
        self.pipe = None
        self.model_loaded = False
        self.load_model()
    
    def load_model(self):
        """Load and initialize the FLUX.1-Kontext model with memory optimization"""
        try:
            print("Loading FLUX.1-Kontext model...")
            
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
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Please ensure you have:")
            print("1. Latest diffusers: pip install -U git+https://github.com/huggingface/diffusers.git")
            print("2. TorchAO: pip install torchao --extra-index-url https://download.pytorch.org/whl/cu121")
            print("3. Access to the FLUX.1-Kontext-dev model on Hugging Face")
            self.model_loaded = False
    
    def get_lighting_prompts(self):
        return {
            "Golden Hour": "the same room with warm golden hour lighting streaming through windows, soft golden glow, warm ambient light",
            "Night Interior": "the same room at night with warm interior lighting, cozy lamps and overhead lights illuminated, dark outside windows",
            "Bright Evening": "the same room in the evening with all interior lights on, bright ceiling LEDs (white ceiling LEDs ) and lamps illuminating every corner, well-lit floor and furniture, dark evening sky visible through windows, cozy indoor ambiance",
            "Moonlit Night": "the same room at midnight with moonlight streaming through windows, cool blue moonlight, minimal interior lighting",
            "Bright Daylight": "the same room with bright natural daylight, sunny day with clear windows, well-lit interior",
            "Cloudy Day": "the same room with gloomy diffused natural lighting from overcast cloudy sky visible through windows,not too dark interior, soft darkness, even grey daylight illumination, objects clearly visible but muted colors, gloomy but not dark interior, existing lights off during daytime",

            "Rainy Day": "the same room during heavy rain with soft grey lighting, raindrops streaming down windows, wet glass surfaces, muted natural light creating a cozy indoor atmosphere, reflections of interior lights on rain-covered windows",
            "Sunset": "the same room during sunset with warm orange and pink light streaming through windows",
            "Dawn": "the same room at dawn with soft morning light, gentle sunrise glow through windows, peaceful atmosphere",
            "Candlelit": "the same room lit by candles and warm ambient lighting, romantic candlelit atmosphere, soft flickering light",
            "Modern LED": "the same room with modern LED lighting, clean bright white light, contemporary lighting design",
            "Vintage Warm": "the same room with vintage warm lighting, retro bulbs and warm yellow light, nostalgic atmosphere",
            "Dramatic Spotlights": "the same room with dramatic spotlighting, theatrical lighting with strong shadows and highlights"
            }
        
    def relight_room(self, input_image, lighting_condition, custom_prompt, guidance_scale, num_inference_steps):
        """Apply relighting to the input room image"""
        if not self.model_loaded:
            return None, "❌ Model not loaded. Please check the setup instructions."
        
        if input_image is None:
            return None, "❌ Please upload an image first."
        
        try:
            # Prepare the input image
            if isinstance(input_image, np.ndarray):
                input_image = Image.fromarray(input_image)
            
            w, h = input_image.size
            
            # Get the appropriate prompt
            lighting_prompts = self.get_lighting_prompts()
            
            if lighting_condition == "Custom" and custom_prompt:
                prompt = f"the same room but {custom_prompt}"
            elif lighting_condition in lighting_prompts:
                prompt = lighting_prompts[lighting_condition]
            else:
                prompt = "the same room with different lighting"
            
            print(f"Generating with prompt: {prompt}")
            
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
            success_message = f"✅ Successfully relit the room with {lighting_condition} lighting!"
            
            return output_image, success_message
            
        except Exception as e:
            error_message = f"❌ Error during generation: {str(e)}"
            print(error_message)
            return None, error_message
    
    def create_interface(self):
        """Create the Gradio interface"""
        
        # Custom CSS for better styling
        css = """
        .gradio-container {
            max-width: 1200px !important;
        }
        .image-container {
            height: 400px !important;
        }
        .lighting-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin: 10px 0;
        }
        """
        
        with gr.Blocks(css=css, title="Room Relighting with FLUX.1-Kontext") as interface:
            gr.Markdown("""
            # 🏠✨ Room Relighting with FLUX.1-Kontext
            
            Transform your room images with different lighting conditions using AI. Upload an image of any room and see it rendered with various lighting scenarios - from golden hour to moonlit nights!
            
            **Features:**
            - 13+ predefined lighting conditions
            - Custom lighting prompts
            - Optimized for <16GB VRAM
            - High-quality image editing
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📸 Input")
                    input_image = gr.Image(
                        label="Upload Room Image",
                        type="pil",
                        height=300
                    )
                    
                    gr.Markdown("### ⚙️ Settings")
                    lighting_condition = gr.Dropdown(
                        choices=["Golden Hour", "Night Interior", "Bright Evening", "Moonlit Night", "Bright Daylight", 
                                "Cloudy Day", "Rainy Day", "Sunset", "Dawn", "Candlelit", 
                                "Modern LED", "Vintage Warm", "Dramatic Spotlights", "Custom"],
                        value="Golden Hour",
                        label="Lighting Condition"
                    )
                    
                    custom_prompt = gr.Textbox(
                        label="Custom Lighting Description",
                        placeholder="e.g., 'with neon lights and futuristic ambiance'",
                        visible=False
                    )
                    
                    with gr.Row():
                        guidance_scale = gr.Slider(
                            minimum=1.0,
                            maximum=5.0,
                            value=2.5,
                            step=0.1,
                            label="Guidance Scale"
                        )
                        
                        num_inference_steps = gr.Slider(
                            minimum=10,
                            maximum=50,
                            value=20,
                            step=5,
                            label="Inference Steps"
                        )
                    
                    generate_btn = gr.Button("🎨 Generate Relit Room", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 🎨 Output")
                    output_image = gr.Image(
                        label="Relit Room",
                        height=300
                    )
                    
                    status_message = gr.Textbox(
                        label="Status",
                        interactive=False,
                        value="Ready to relight your room! Upload an image and click generate."
                    )
            
            # Show/hide custom prompt based on selection
            def toggle_custom_prompt(choice):
                return gr.update(visible=(choice == "Custom"))
            
            lighting_condition.change(
                toggle_custom_prompt,
                inputs=[lighting_condition],
                outputs=[custom_prompt]
            )
            
            # Generate button action
            generate_btn.click(
                self.relight_room,
                inputs=[input_image, lighting_condition, custom_prompt, guidance_scale, num_inference_steps],
                outputs=[output_image, status_message]
            )
            
            # Examples section
            gr.Markdown("""
            ### 💡 Tips for Best Results:
            - Use high-quality room images with clear details
            - Interior shots work better than exterior shots  
            - Images with visible windows show more dramatic lighting changes
            - Try different guidance scales (2.0-3.5 typically work well)
            - Higher inference steps = better quality but slower generation
            
            ### 🔧 Setup Requirements:
            ```bash
            pip install -U git+https://github.com/huggingface/diffusers.git
            pip install torchao --extra-index-url https://download.pytorch.org/whl/cu121
            ```
            
            You'll also need access to the FLUX.1-Kontext-dev model on Hugging Face.
            """)
            
            # Example images (you can add these if you have sample room images)
            gr.Examples(
                examples=[
                    ["Golden Hour", "Warm golden sunlight streaming through large windows"],
                    ["Night Interior", "Cozy evening lighting with lamps"],
                    ["Bright Evening", "All interior lights on with well-lit floor and furniture"],
                    ["Moonlit Night", "Cool blue moonlight with minimal interior lighting"],
                    ["Rainy Day", "Soft grey light with rain visible through windows"],
                    ["Custom", "with colorful neon lights and cyberpunk atmosphere"]
                ],
                inputs=[lighting_condition, custom_prompt],
                label="Example Lighting Scenarios"
            )
        
        return interface

# Initialize the app
app = RoomRelightingApp()

# Create and launch the interface
if __name__ == "__main__":
    interface = app.create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7600,
        share=True,
        debug=True
    )