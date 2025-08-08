import gradio as gr
import plotly.graph_objects as go
import torch
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.model_handler import ModelHandler
from core.attention import AttentionProcessor
from core.cache import AttentionCache
from visualization.plotly_viz import AttentionVisualizer
from config import Config

class TokenVisualizerApp:
    def __init__(self):
        self.config = Config()
        self.model_handler = ModelHandler()
        self.cache = AttentionCache(max_size=self.config.CACHE_SIZE)
        self.visualizer = AttentionVisualizer(self.config)
        self.current_data = None
        self.current_figure = None
        self.model_loaded = False
        
    def load_model(self, model_name: str = None) -> str:
        """Load the model and return status message."""
        if not model_name:
            model_name = self.config.DEFAULT_MODEL
        
        success, message = self.model_handler.load_model(model_name)
        self.model_loaded = success
        
        if success:
            model_info = self.model_handler.get_model_info()
            return f"✅ Model loaded: {model_name}\n📊 Parameters: {model_info['num_parameters']:,}\n🖥️ Device: {model_info['device']}"
        else:
            return f"❌ Failed to load model: {message}"
    
    def generate_and_visualize(
        self,
        prompt: str,
        max_tokens: int,
        threshold: float,
        temperature: float,
        normalization: str,
        progress=gr.Progress()
    ):
        """Main generation and visualization function."""
        if not self.model_loaded:
            return None, "Please load a model first!", None
        
        if not prompt.strip():
            return None, "Please enter a prompt!", None
        
        progress(0.2, desc="Checking cache...")
        
        # Check cache
        cache_key = self.cache.get_key(
            prompt, max_tokens, 
            self.model_handler.model_name, 
            temperature
        )
        cached = self.cache.get(cache_key)
        
        if cached:
            progress(0.5, desc="Using cached data...")
            self.current_data = cached
        else:
            progress(0.3, desc="Generating text...")
            
            # Generate new
            attention_data, output_tokens, input_tokens, generated_text = \
                self.model_handler.generate_with_attention(
                    prompt, max_tokens, temperature
                )
            
            if attention_data is None:
                return None, f"Generation failed: {generated_text}", None
            
            progress(0.6, desc="Processing attention...")
            
            # Process attention based on normalization method
            if normalization == "separate":
                attention_matrices = AttentionProcessor.process_attention_separate(
                    attention_data, input_tokens, output_tokens
                )
            else:
                attention_matrices = AttentionProcessor.process_attention_joint(
                    attention_data, input_tokens, output_tokens
                )
            
            self.current_data = {
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'attention_matrices': attention_matrices,
                'generated_text': generated_text,
                'attention_data': attention_data  # Keep raw for step updates
            }
            
            # Cache it
            self.cache.set(cache_key, self.current_data)
        
        progress(0.8, desc="Creating visualization...")
        
        # Create visualization
        self.current_figure = self.visualizer.create_interactive_plot(
            self.current_data['input_tokens'],
            self.current_data['output_tokens'],
            self.current_data['attention_matrices'],
            threshold=threshold,
            initial_step=len(self.current_data['output_tokens']) - 1,
            normalization=normalization
        )
        
        progress(1.0, desc="Complete!")
        
        # Create info text
        info_text = f"📝 Generated: {self.current_data['generated_text']}\n"
        info_text += f"🔤 Input tokens: {len(self.current_data['input_tokens'])}\n"
        info_text += f"🔤 Output tokens: {len(self.current_data['output_tokens'])}"
        
        # Update step slider maximum
        max_step = len(self.current_data['output_tokens']) - 1
        
        return self.current_figure, info_text, gr.update(maximum=max_step, value=max_step)
    
    def update_step(self, step_idx: int, threshold: float):
        """Update visualization for different step."""
        if not self.current_data or not self.current_figure:
            return self.current_figure
        
        updated_fig = self.visualizer.update_for_step(
            self.current_figure,
            step_idx,
            self.current_data['attention_matrices'],
            self.current_data['output_tokens'],
            threshold
        )
        
        return updated_fig
    
    def update_threshold(self, threshold: float, normalization: str):
        """Update threshold and refresh visualization."""
        if not self.current_data:
            return self.current_figure
        
        # Recreate visualization with new threshold
        step = self.visualizer.current_state['current_step']
        self.current_figure = self.visualizer.create_interactive_plot(
            self.current_data['input_tokens'],
            self.current_data['output_tokens'],
            self.current_data['attention_matrices'],
            threshold=threshold,
            initial_step=step,
            normalization=normalization
        )
        
        return self.current_figure
    
    def filter_token_connections(self, token_idx: int, token_type: str, threshold: float):
        """Filter connections for selected token."""
        if not self.current_data or not self.current_figure:
            return self.current_figure
        
        updated_fig = self.visualizer.filter_by_token(
            self.current_figure,
            token_idx,
            token_type,
            self.current_data['attention_matrices'],
            threshold
        )
        
        return updated_fig
    
    def reset_view(self, threshold: float):
        """Reset to show all connections."""
        if not self.current_data or not self.current_figure:
            return self.current_figure
        
        updated_fig = self.visualizer.show_all_connections(
            self.current_figure,
            self.current_data['attention_matrices'],
            threshold
        )
        
        return updated_fig

def create_gradio_interface():
    """Create the Gradio interface."""
    app = TokenVisualizerApp()
    
    with gr.Blocks(
        title="Token Attention Visualizer",
        theme=gr.themes.Soft(),
        css="""
        .container { max-width: 1400px; margin: auto; }
        .plot-container { min-height: 600px; }
        """
    ) as demo:
        gr.Markdown(
            """
            # 🔍 Token Attention Visualizer
            
            Interactive visualization of attention patterns in Large Language Models.
            See how tokens attend to each other during text generation!
            """
        )
        
        # Model loading section
        with gr.Row():
            with gr.Column(scale=3):
                model_input = gr.Textbox(
                    label="Model Name",
                    value=app.config.DEFAULT_MODEL,
                    placeholder="Enter Hugging Face model name..."
                )
            with gr.Column(scale=1):
                load_model_btn = gr.Button("Load Model", variant="primary")
        
        model_status = gr.Textbox(
            label="Model Status",
            value="No model loaded",
            interactive=False,
            lines=3
        )
        
        gr.Markdown("---")
        
        # Generation controls
        with gr.Row():
            with gr.Column(scale=2):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    value=app.config.DEFAULT_PROMPT,
                    lines=3,
                    placeholder="Enter your prompt here..."
                )
            
            with gr.Column(scale=1):
                max_tokens_input = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=app.config.DEFAULT_MAX_TOKENS,
                    step=1,
                    label="Max Tokens to Generate"
                )
                
                temperature_input = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=app.config.DEFAULT_TEMPERATURE,
                    step=0.1,
                    label="Temperature"
                )
                
                normalization_input = gr.Radio(
                    choices=["separate", "joint"],
                    value="separate",
                    label="Normalization Method",
                    info="Separate: normalize input/output independently | Joint: normalize together"
                )
        
        generate_btn = gr.Button("🚀 Generate & Visualize", variant="primary", size="lg")
        
        # Results section
        with gr.Row():
            generated_info = gr.Textbox(
                label="Generation Info",
                interactive=False,
                lines=3
            )
        
        # Visualization controls
        with gr.Row():
            with gr.Column(scale=1):
                threshold_input = gr.Slider(
                    minimum=0.0,
                    maximum=0.2,
                    value=app.config.DEFAULT_THRESHOLD,
                    step=0.001,
                    label="Attention Threshold",
                    info="Hide connections below this weight"
                )
            
            with gr.Column(scale=1):
                step_slider = gr.Slider(
                    minimum=0,
                    maximum=20,
                    value=0,
                    step=1,
                    label="Generation Step",
                    info="Navigate through generation steps"
                )
            
            with gr.Column(scale=1):
                with gr.Row():
                    reset_btn = gr.Button("🔄 Reset View", size="sm")
                    # Token selection would be handled through plot clicks
        
        # Main visualization
        plot_output = gr.Plot(
            label="Attention Visualization",
            elem_classes="plot-container"
        )
        
        # Instructions
        with gr.Accordion("📖 How to Use", open=False):
            gr.Markdown(
                """
                ### Instructions:
                1. **Load a model** from Hugging Face (default: Llama-3.2-1B)
                2. **Enter a prompt** and configure generation settings
                3. **Click Generate** to create text and visualize attention
                4. **Interact with the visualization:**
                   - Use the **step slider** to navigate through generation steps
                   - Adjust the **threshold** to filter weak connections
                   - Click on **tokens** in the plot to filter their connections
                   - Click **Reset View** to show all connections
                
                ### Understanding the Visualization:
                - **Blue lines**: Attention from input to output tokens
                - **Orange curves**: Attention between output tokens
                - **Line thickness**: Represents attention weight strength
                - **Node colors**: Blue = input tokens, Coral = generated tokens
                """
            )
        
        # Event handlers
        load_model_btn.click(
            fn=app.load_model,
            inputs=[model_input],
            outputs=[model_status]
        )
        
        generate_btn.click(
            fn=app.generate_and_visualize,
            inputs=[
                prompt_input,
                max_tokens_input,
                threshold_input,
                temperature_input,
                normalization_input
            ],
            outputs=[plot_output, generated_info, step_slider]
        )
        
        step_slider.change(
            fn=app.update_step,
            inputs=[step_slider, threshold_input],
            outputs=[plot_output]
        )
        
        threshold_input.change(
            fn=app.update_threshold,
            inputs=[threshold_input, normalization_input],
            outputs=[plot_output]
        )
        
        reset_btn.click(
            fn=app.reset_view,
            inputs=[threshold_input],
            outputs=[plot_output]
        )
        
        # Load default model on startup
        demo.load(
            fn=app.load_model,
            inputs=[gr.State(app.config.DEFAULT_MODEL)],
            outputs=[model_status]
        )
    
    return demo

if __name__ == "__main__":
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ CUDA not available, using CPU")
    
    # Create and launch the app
    demo = create_gradio_interface()
    demo.launch(
        share=False,  # Set to True for public URL
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,  # Default Gradio port
        inbrowser=False  # Don't auto-open browser
    )