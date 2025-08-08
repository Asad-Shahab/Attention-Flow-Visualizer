import gradio as gr
import plotly.graph_objects as go
import torch
import sys
import os
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.model_handler import ModelHandler
from core.attention import AttentionProcessor
from core.cache import AttentionCache
from visualization.plotly_viz import AttentionVisualizer
from visualization.d3_viz import D3AttentionVisualizer
from visualization.simple_svg_viz import SimpleSVGVisualizer
from config import Config

class TokenVisualizerApp:
    def __init__(self):
        self.config = Config()
        self.model_handler = ModelHandler(config=self.config)
        self.cache = AttentionCache(max_size=self.config.CACHE_SIZE)
        self.visualizer = AttentionVisualizer(self.config)
        self.d3_visualizer = D3AttentionVisualizer(self.config)
        self.svg_visualizer = SimpleSVGVisualizer(self.config)
        self.current_data = None
        self.current_figure = None
        self.model_loaded = False
        self.use_d3 = False  # Use simple SVG instead of D3
        self.use_svg = True  # Use simple SVG visualization
        
    def compute_overlay_css(self) -> str:
        """Compute CSS to position overlay buttons over SVG nodes for Simple SVG mode."""
        try:
            if not self.current_data:
                return "<style>#viz_container{position:relative;width:%dpx;height:%dpx;margin:0 auto;}</style>" % (self.config.PLOT_WIDTH, self.config.PLOT_HEIGHT)

            input_tokens = self.current_data['input_tokens']
            output_tokens = self.current_data['output_tokens']
            width = self.config.PLOT_WIDTH
            height = self.config.PLOT_HEIGHT
            margin = 100  # must match simple_svg_viz

            input_x = margin
            output_x = width - margin

            input_y_positions = []
            output_y_positions = []
            if len(input_tokens) > 0:
                input_spacing = (height - 2 * margin) / max(1, len(input_tokens) - 1)
                input_y_positions = [margin + i * input_spacing for i in range(len(input_tokens))]
            if len(output_tokens) > 0:
                output_spacing = (height - 2 * margin) / max(1, len(output_tokens) - 1)
                output_y_positions = [margin + i * output_spacing for i in range(len(output_tokens))]

            css_lines = [
                "<style>",
                f"#viz_container {{ position: relative; width: {width}px; height: {height}px; margin: 0 auto; }}",
                ".overlay-btn { position: absolute; width: 28px; height: 28px; background: transparent; border: none; padding: 0; transform: translate(-50%, -50%); cursor: pointer; }",
            ]

            # Input buttons
            for i in range(64):
                if i < len(input_tokens):
                    y = input_y_positions[i]
                    css_lines.append(f"#btn_in_{i} {{ left: {input_x}px; top: {y}px; display: block; }}")
                else:
                    css_lines.append(f"#btn_in_{i} {{ display: none; }}")

            # Output buttons
            for j in range(64):
                if j < len(output_tokens):
                    y = output_y_positions[j]
                    css_lines.append(f"#btn_out_{j} {{ left: {output_x}px; top: {y}px; display: block; }}")
                else:
                    css_lines.append(f"#btn_out_{j} {{ display: none; }}")

            css_lines.append("</style>")
            return "\n".join(css_lines)
        except Exception:
            return "<style>#viz_container{position:relative;width:%dpx;height:%dpx;margin:0 auto;}</style>" % (self.config.PLOT_WIDTH, self.config.PLOT_HEIGHT)

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
        if self.use_svg:
            # Create simple SVG visualization
            viz_html = self.svg_visualizer.create_visualization_html(
                self.current_data['input_tokens'],
                self.current_data['output_tokens'],
                self.current_data['attention_matrices'],
                threshold=threshold,
                initial_step=len(self.current_data['output_tokens']) - 1
            )
            self.current_figure = viz_html
        elif self.use_d3:
            # Create D3 visualization HTML
            viz_html = self.d3_visualizer.create_visualization_html(
                self.current_data['input_tokens'],
                self.current_data['output_tokens'],
                self.current_data['attention_matrices'],
                threshold=threshold,
                initial_step=len(self.current_data['output_tokens']) - 1
            )
            self.current_figure = viz_html
        else:
            # Create Plotly visualization
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
        
        # Also provide input token choices for selector
        input_choices = list(self.current_data['input_tokens'])
        
        return (
            self.current_figure,
            info_text,
            gr.update(maximum=max_step, value=max_step),
            gr.update(choices=input_choices, value=None)
        )
    
    def update_step(self, step_idx: int, threshold: float):
        """Update visualization for different step."""
        if not self.current_data or not self.current_figure:
            return self.current_figure
        
        if self.use_svg:
            # Recreate SVG visualization for new step
            viz_html = self.svg_visualizer.create_visualization_html(
                self.current_data['input_tokens'],
                self.current_data['output_tokens'],
                self.current_data['attention_matrices'],
                threshold=threshold,
                initial_step=step_idx
            )
            self.current_figure = viz_html
            return viz_html
        elif self.use_d3:
            # Update current step
            self.d3_visualizer.current_state['current_step'] = step_idx
            
            # Get selection state
            selected_token = self.d3_visualizer.current_state.get('selected_token')
            selected_type = self.d3_visualizer.current_state.get('selected_type')
            
            # Recreate D3 visualization for new step
            viz_html = self.d3_visualizer.create_visualization_html(
                self.current_data['input_tokens'],
                self.current_data['output_tokens'],
                self.current_data['attention_matrices'],
                threshold=threshold,
                initial_step=step_idx,
                selected_token=selected_token,
                selected_type=selected_type
            )
            self.current_figure = viz_html
            return viz_html
        else:
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
        
        if self.use_svg or self.use_d3:
            # Get current state
            current_step = self.d3_visualizer.current_state.get('current_step', len(self.current_data['output_tokens']) - 1)
            selected_token = self.d3_visualizer.current_state.get('selected_token')
            selected_type = self.d3_visualizer.current_state.get('selected_type')
            
            # Recreate D3 visualization with new threshold
            viz_html = self.d3_visualizer.create_visualization_html(
                self.current_data['input_tokens'],
                self.current_data['output_tokens'],
                self.current_data['attention_matrices'],
                threshold=threshold,
                initial_step=current_step,
                selected_token=selected_token,
                selected_type=selected_type
            )
            self.current_figure = viz_html
            return viz_html
        else:
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
        
        if self.use_svg or self.use_d3:
            # Get current state
            current_step = self.d3_visualizer.current_state.get('current_step', len(self.current_data['output_tokens']) - 1)
            
            # Reset state
            self.d3_visualizer.current_state['selected_token'] = None
            self.d3_visualizer.current_state['selected_type'] = None
            
            # Recreate D3 visualization without selection
            viz_html = self.d3_visualizer.create_visualization_html(
                self.current_data['input_tokens'],
                self.current_data['output_tokens'],
                self.current_data['attention_matrices'],
                threshold=threshold,
                initial_step=current_step
            )
            self.current_figure = viz_html
            return viz_html
        else:
            updated_fig = self.visualizer.show_all_connections(
                self.current_figure,
                self.current_data['attention_matrices'],
                threshold
            )
            return updated_fig

    def on_d3_token_click(self, click_data: str, threshold: float):
        """Handle token clicks from visualization."""
        if not self.current_data or not self.current_figure or not click_data:
            return self.current_figure, gr.update()
        
        try:
            data = json.loads(click_data)
            token_idx = data.get('index')
            token_type = data.get('type')
            
            if token_idx is None or token_type is None:
                return self.current_figure, gr.update()
            
            # Store state
            self._current_step = getattr(self, '_current_step', len(self.current_data['output_tokens']) - 1)
            self._selected_token = token_idx
            self._selected_type = token_type
            
            if self.use_svg:
                # Recreate SVG visualization with selected token
                viz_html = self.svg_visualizer.create_visualization_html(
                    self.current_data['input_tokens'],
                    self.current_data['output_tokens'],
                    self.current_data['attention_matrices'],
                    threshold=threshold,
                    initial_step=self._current_step,
                    selected_token=token_idx,
                    selected_type=token_type
                )
                self.current_figure = viz_html
            elif self.use_d3:
                # Get current step
                current_step = self.d3_visualizer.current_state.get('current_step', len(self.current_data['output_tokens']) - 1)
                
                # Recreate D3 visualization with selected token
                viz_html = self.d3_visualizer.create_visualization_html(
                    self.current_data['input_tokens'],
                    self.current_data['output_tokens'],
                    self.current_data['attention_matrices'],
                    threshold=threshold,
                    initial_step=current_step,
                    selected_token=token_idx,
                    selected_type=token_type
                )
                self.current_figure = viz_html
                self.d3_visualizer.current_state['selected_token'] = token_idx
                self.d3_visualizer.current_state['selected_type'] = token_type
            
            # Update dropdown if input token was clicked
            if token_type == 'input':
                return self.current_figure, gr.update(value=self.current_data['input_tokens'][token_idx])
            else:
                return self.current_figure, gr.update()
                
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Error in token click handler: {e}")
            return self.current_figure, gr.update()
    
    def on_input_token_select(self, token_label: str, threshold: float):
        """Filter connections when an input token is selected from dropdown."""
        if not self.current_data or not self.current_figure:
            return self.current_figure
        if token_label is None or token_label == "":
            return self.current_figure
        try:
            token_idx = self.current_data['input_tokens'].index(token_label)
        except ValueError:
            return self.current_figure
        
        if self.use_svg or self.use_d3:
            # Recreate D3 visualization with selected token
            current_step = self.d3_visualizer.current_state.get('current_step', len(self.current_data['output_tokens']) - 1)
            viz_html = self.d3_visualizer.create_visualization_html(
                self.current_data['input_tokens'],
                self.current_data['output_tokens'],
                self.current_data['attention_matrices'],
                threshold=threshold,
                initial_step=current_step,
                selected_token=token_idx,
                selected_type='input'
            )
            self.current_figure = viz_html
            return viz_html
        else:
            return self.visualizer.filter_by_token(
                self.current_figure,
                token_idx,
                'input',
                self.current_data['attention_matrices'],
                threshold
            )


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
            with gr.Column(scale=1):
                input_token_selector = gr.Dropdown(
                    label="Filter by Input Token",
                    choices=[],
                    interactive=True,
                    allow_custom_value=False
                )
        
        # Main visualization - use HTML for D3
        viz_output = gr.HTML(
            label="Attention Visualization",
            elem_classes="viz-container"
        )

        # Overlay container: CSS and invisible buttons positioned over SVG nodes (Simple SVG path)
        with gr.Column(elem_id="viz_container"):
            # CSS placeholder that we'll update after generation to position the buttons
            overlay_css = gr.HTML(value="", elem_id="viz_overlay_css")

            # Pre-create a fixed number of buttons for inputs/outputs; we'll show/hide via CSS
            max_input_buttons = 64
            max_output_buttons = 64
            input_buttons = []
            output_buttons = []
            for _i in range(max_input_buttons):
                input_buttons.append(gr.Button(" ", elem_id=f"btn_in_{_i}", visible=True, size="sm"))
            for _j in range(max_output_buttons):
                output_buttons.append(gr.Button(" ", elem_id=f"btn_out_{_j}", visible=True, size="sm"))
            reset_btn_overlay = gr.Button("Reset View (Overlay)", elem_id="btn_reset_overlay")
        
        # Hidden textbox for D3 click events
        clicked_token_d3 = gr.Textbox(
            visible=False, 
            elem_id="clicked-token-d3"
        )
        
        debug_info = gr.HTML(
            """<div style="font-size: 12px; color: #555; margin-top: 10px;">
            Click on any token to filter connections. D3.js visualization active.
            </div>"""
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
        
        def _generate_and_css(prompt, max_tokens, threshold, temperature, normalization):
            fig, info, step, dropdown = app.generate_and_visualize(
                prompt, max_tokens, threshold, temperature, normalization
            )
            css = app.compute_overlay_css()
            return fig, info, step, dropdown, css

        generate_btn.click(
            fn=_generate_and_css,
            inputs=[
                prompt_input,
                max_tokens_input,
                threshold_input,
                temperature_input,
                normalization_input
            ],
            outputs=[viz_output, generated_info, step_slider, input_token_selector, overlay_css]
        )
        
        def _update_step_and_css(step_value, threshold):
            fig = app.update_step(step_value, threshold)
            css = app.compute_overlay_css()
            return fig, css

        step_slider.change(
            fn=_update_step_and_css,
            inputs=[step_slider, threshold_input],
            outputs=[viz_output, overlay_css]
        )
        
        def _update_threshold_and_css(threshold, normalization):
            fig = app.update_threshold(threshold, normalization)
            css = app.compute_overlay_css()
            return fig, css

        threshold_input.change(
            fn=_update_threshold_and_css,
            inputs=[threshold_input, normalization_input],
            outputs=[viz_output, overlay_css]
        )
        
        def _reset_and_css(threshold):
            fig = app.reset_view(threshold)
            css = app.compute_overlay_css()
            return fig, css

        reset_btn.click(
            fn=_reset_and_css,
            inputs=[threshold_input],
            outputs=[viz_output, overlay_css]
        )

        # Overlay button click handlers (map to the same filtering path)
        def _click_input_factory(idx: int):
            return lambda thr: app.on_d3_token_click(json.dumps({"index": idx, "type": "input"}), thr)

        def _click_output_factory(idx: int):
            return lambda thr: app.on_d3_token_click(json.dumps({"index": idx, "type": "output"}), thr)

        for _i, _btn in enumerate(input_buttons):
            _btn.click(
                fn=_click_input_factory(_i),
                inputs=[threshold_input],
                outputs=[viz_output, input_token_selector]
            )

        for _j, _btn in enumerate(output_buttons):
            _btn.click(
                fn=_click_output_factory(_j),
                inputs=[threshold_input],
                outputs=[viz_output, input_token_selector]
            )

        reset_btn_overlay.click(
            fn=app.reset_view,
            inputs=[threshold_input],
            outputs=[viz_output]
        )

        # Dropdown selection to filter connections by input token
        def _on_input_select_and_css(label, threshold):
            fig = app.on_input_token_select(label, threshold)
            css = app.compute_overlay_css()
            return fig, css

        input_token_selector.change(
            fn=_on_input_select_and_css,
            inputs=[input_token_selector, threshold_input],
            outputs=[viz_output, overlay_css]
        )
        
        # D3 click handler
        clicked_token_d3.change(
            fn=app.on_d3_token_click,
            inputs=[clicked_token_d3, threshold_input],
            outputs=[viz_output, input_token_selector]
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