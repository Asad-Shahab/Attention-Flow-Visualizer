import gradio as gr
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
from config import Config
from visualization.d3_viz import create_d3_visualization

class TokenVisualizerApp:
    def __init__(self):
        self.config = Config()
        self.model_handler = ModelHandler(config=self.config)
        self.cache = AttentionCache(max_size=self.config.CACHE_SIZE)
        self.current_data = None
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
        """Main generation function (no visualization)."""
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
        
        progress(1.0, desc="Complete!")
        
        # Create info text
        info_text = f"📝 Generated: {self.current_data['generated_text']}\n"
        info_text += f"🔤 Input tokens: {len(self.current_data['input_tokens'])}\n"
        info_text += f"🔤 Output tokens: {len(self.current_data['output_tokens'])}"
        
        return (
            info_text,
        )
    
    def update_step(self, step_idx: int, threshold: float):
        """No-op placeholder after removing visualization."""
        return None
    
    def update_threshold(self, threshold: float, normalization: str):
        """No-op placeholder after removing visualization."""
        return None
    
    def filter_token_connections(self, token_idx: int, token_type: str, threshold: float):
        """Removed visualization; keep placeholder."""
        return None
    
    def reset_view(self, threshold: float):
        """Removed visualization; keep placeholder."""
        return None

    def on_d3_token_click(self, click_data: str, threshold: float):
        """Removed visualization; keep placeholder for compatibility."""
        return None, gr.update()
    
    def on_input_token_select(self, token_label: str, threshold: float):
        """Removed visualization; keep placeholder for compatibility."""
        return None

    def prepare_d3_data(self, step_idx: int, threshold: float = 0.01, filter_token: str = None):
        """
        Convert attention data to D3.js-friendly JSON format.
        
        Args:
            step_idx: Generation step to visualize (0-based)
            threshold: Minimum attention weight to include
            filter_token: Token to filter by (format: "[IN] token" or "[OUT] token" or "All tokens")
            
        Returns:
            dict: JSON structure with nodes and links for D3.js
        """
        if not self.current_data:
            return {"nodes": [], "links": []}
        
        input_tokens = self.current_data['input_tokens']
        output_tokens = self.current_data['output_tokens']
        attention_matrices = self.current_data['attention_matrices']
        
        # Ensure step_idx is within bounds
        if step_idx >= len(attention_matrices):
            step_idx = len(attention_matrices) - 1
        
        attention_matrix = attention_matrices[step_idx]
        
        # Create nodes
        nodes = []
        
        # Add input nodes
        for i, token in enumerate(input_tokens):
            nodes.append({
                "id": f"input_{i}",
                "token": token,
                "type": "input",
                "index": i
            })
        
        # Add output nodes (up to current step)
        for i in range(step_idx + 1):
            if i < len(output_tokens):
                nodes.append({
                    "id": f"output_{i}",
                    "token": output_tokens[i],
                    "type": "output",
                    "index": i
                })
        
        # Parse filter token
        filter_type = None
        filter_idx = None
        if filter_token and filter_token != "All tokens":
            if filter_token.startswith("[IN] "):
                filter_type = "input"
                filter_token_text = filter_token[5:]  # Remove "[IN] " prefix
                filter_idx = next((i for i, token in enumerate(input_tokens) if token == filter_token_text), None)
            elif filter_token.startswith("[OUT] "):
                filter_type = "output"
                filter_token_text = filter_token[6:]  # Remove "[OUT] " prefix
                filter_idx = next((i for i, token in enumerate(output_tokens) if token == filter_token_text), None)
        
        # Create links from attention matrices - show ALL steps up to current step
        links = []
        
        # Show connections for all steps up to and including step_idx
        for current_step in range(step_idx + 1):
            if current_step < len(attention_matrices):
                step_attention = attention_matrices[current_step]
                
                # Links from input tokens to this output token
                input_attention = step_attention['input_attention']
                if input_attention is not None:
                    for input_idx in range(len(input_tokens)):
                        if input_idx < len(input_attention):  # Check bounds
                            weight = float(input_attention[input_idx])
                            if weight >= threshold:
                                # Apply filtering
                                show_link = True
                                if filter_type == "input" and filter_idx is not None:
                                    # Only show connections involving the selected input token
                                    show_link = (input_idx == filter_idx)
                                elif filter_type == "output" and filter_idx is not None:
                                    # Only show connections involving the selected output token
                                    show_link = (current_step == filter_idx)
                                
                                if show_link:
                                    links.append({
                                        "source": f"input_{input_idx}",
                                        "target": f"output_{current_step}",
                                        "weight": weight,
                                        "type": "input_to_output"
                                    })
                
                # Links from previous output tokens to this output token
                output_attention = step_attention['output_attention']
                if output_attention is not None and current_step > 0:
                    for prev_output_idx in range(current_step):
                        if prev_output_idx < len(output_attention):  # Check bounds
                            weight = float(output_attention[prev_output_idx])
                            if weight >= threshold:
                                # Apply filtering
                                show_link = True
                                if filter_type == "input" and filter_idx is not None:
                                    # Don't show output-to-output connections when filtering by input
                                    show_link = False
                                elif filter_type == "output" and filter_idx is not None:
                                    # Only show connections involving the selected output token
                                    show_link = (prev_output_idx == filter_idx or current_step == filter_idx)
                                
                                if show_link:
                                    links.append({
                                        "source": f"output_{prev_output_idx}",
                                        "target": f"output_{current_step}",
                                        "weight": weight,
                                        "type": "output_to_output"
                                    })
        
        return {
            "nodes": nodes,
            "links": links,
            "step": step_idx,
            "total_steps": len(attention_matrices),
            "input_count": len(input_tokens),
            "output_count": step_idx + 1
        }

    def create_d3_visualization_html(self, step_idx: int = 0, threshold: float = 0.01, filter_token: str = None):
        """
        Create D3.js visualization HTML for the current data.
        
        Args:
            step_idx: Generation step to visualize (0-based)
            threshold: Minimum attention weight to include
            filter_token: Token to filter by (format: "[IN] token" or "[OUT] token")
            
        Returns:
            str: HTML string for D3.js visualization
        """
        if not self.current_data:
            return "<div>No data available. Generate text first!</div>"
        
        d3_data = self.prepare_d3_data(step_idx, threshold, filter_token)
        
        viz_html = create_d3_visualization(d3_data)
        return viz_html

    def get_token_choices(self):
        """
        Get list of token choices for dropdown.
        
        Returns:
            list: List of token strings for dropdown options
        """
        if not self.current_data:
            return []
        
        input_tokens = self.current_data['input_tokens']
        output_tokens = self.current_data['output_tokens']
        
        # Create choices with prefixes to distinguish input/output
        choices = ["All tokens"]
        choices.extend([f"[IN] {token}" for token in input_tokens])
        choices.extend([f"[OUT] {token}" for token in output_tokens])
        
        return choices


def create_gradio_interface():
    """Create the Gradio interface."""
    app = TokenVisualizerApp()
    
    with gr.Blocks(
        title="Token Attention Visualizer",
        css="""
        /* Default/Light mode styles */
        .main-header {
            text-align: center;
            padding: 2rem 0 3rem 0;
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            border-radius: 1rem;
            margin-bottom: 2rem;
            border: 1px solid #e2e8f0;
        }
        
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1e293b;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #1e293b 0%, #3b82f6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .main-subtitle {
            font-size: 1.125rem;
            color: #64748b;
            font-weight: 400;
        }
        
        .section-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e2e8f0;
        }
        
        /* Explicit light mode overrides */
        .light .main-header,
        [data-theme="light"] .main-header {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            border: 1px solid #e2e8f0;
        }
        
        .light .main-title,
        [data-theme="light"] .main-title {
            color: #1e293b;
            background: linear-gradient(135deg, #1e293b 0%, #3b82f6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .light .main-subtitle,
        [data-theme="light"] .main-subtitle {
            color: #64748b;
        }
        
        .light .section-title,
        [data-theme="light"] .section-title {
            color: #1e293b;
            border-bottom: 2px solid #e2e8f0;
        }
        
        /* Dark mode styles with higher specificity */
        .dark .main-header,
        [data-theme="dark"] .main-header {
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%) !important;
            border: 1px solid #475569 !important;
        }
        
        .dark .main-title,
        [data-theme="dark"] .main-title {
            color: #f1f5f9 !important;
            background: linear-gradient(135deg, #f1f5f9 0%, #60a5fa 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
        }
        
        .dark .main-subtitle,
        [data-theme="dark"] .main-subtitle {
            color: #cbd5e1 !important;
        }
        
        .dark .section-title,
        [data-theme="dark"] .section-title {
            color: #f1f5f9 !important;
            border-bottom: 2px solid #475569 !important;
        }
        
        /* System dark mode - only apply when no explicit theme is set */
        @media (prefers-color-scheme: dark) {
            :root:not([data-theme="light"]) .main-header {
                background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
                border: 1px solid #475569;
            }
            
            :root:not([data-theme="light"]) .main-title {
                color: #f1f5f9;
                background: linear-gradient(135deg, #f1f5f9 0%, #60a5fa 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            
            :root:not([data-theme="light"]) .main-subtitle {
                color: #cbd5e1;
            }
            
            :root:not([data-theme="light"]) .section-title {
                color: #f1f5f9;
                border-bottom: 2px solid #475569;
            }
        }
        
        .load-model-btn {
            background: linear-gradient(135deg, #f97316 0%, #ea580c 100%) !important;
            color: white !important;
            border: none !important;
            font-weight: 600 !important;
            padding: 0.75rem 2rem !important;
            border-radius: 0.5rem !important;
            box-shadow: 0 4px 6px -1px rgba(249, 115, 22, 0.25) !important;
            transition: all 0.2s ease !important;
        }
        
        .load-model-btn:hover {
            background: linear-gradient(135deg, #ea580c 0%, #dc2626 100%) !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 8px -1px rgba(249, 115, 22, 0.35) !important;
        }
        """
    ) as demo:
        gr.HTML("""
            <div class="main-header">
                <h1 class="main-title">Token Attention Visualizer</h1>
                <p class="main-subtitle">Interactive visualization of attention patterns in Large Language Models</p>
            </div>
        """)
        
        with gr.Row():
            # Left Panel - Controls
            with gr.Column(scale=1):
                gr.HTML('<h2 class="section-title">Model & Generation</h2>')
                
                # Model loading
                model_input = gr.Textbox(
                    label="Model Name",
                    value=app.config.DEFAULT_MODEL,
                    placeholder="Enter Hugging Face model name..."
                )
                load_model_btn = gr.Button("Load Model", variant="primary", elem_classes=["load-model-btn"])
                
                model_status = gr.Textbox(
                    label="Model Status",
                    value="No model loaded",
                    interactive=False,
                    lines=2
                )
                
                # Generation controls
                prompt_input = gr.Textbox(
                    label="Prompt",
                    value=app.config.DEFAULT_PROMPT,
                    lines=3,
                    placeholder="Enter your prompt here..."
                )
                
                max_tokens_input = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=app.config.DEFAULT_MAX_TOKENS,
                    step=1,
                    label="Max Tokens"
                )
                
                temperature_input = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=app.config.DEFAULT_TEMPERATURE,
                    step=0.1,
                    label="Temperature"
                )
                
                generate_btn = gr.Button("Generate", variant="primary", size="lg")
                
                generated_info = gr.Textbox(
                    label="Generation Info",
                    interactive=False,
                    lines=4
                )
                
                gr.HTML('<h2 class="section-title">Visualization Controls</h2>')
                
                step_slider = gr.Slider(
                    minimum=0,
                    maximum=10,
                    value=0,
                    step=1,
                    label="Generation Step",
                    info="Navigate through generation steps"
                )
                
                threshold_slider = gr.Slider(
                    minimum=0.001,
                    maximum=0.5,
                    value=0.01,
                    step=0.001,
                    label="Attention Threshold", 
                    info="Filter weak connections"
                )
                
                token_dropdown = gr.Dropdown(
                    choices=["All tokens"],
                    value="All tokens",
                    label="Filter by Token",
                    info="Select a token to highlight"
                )
            
            # Right Panel - Visualization
            with gr.Column(scale=2):
                gr.HTML('<h2 class="section-title">Attention Visualization</h2>')
                
                d3_visualization = gr.HTML(
                    value="""<div style='height: 700px; display: flex; align-items: center; justify-content: center; font-size: 16px;'>
                        <div style='text-align: center;'>
                            <div style='font-size: 3rem; margin-bottom: 16px; opacity: 0.5;'>⚪</div>
                            <div style='font-weight: 500; margin-bottom: 8px;'>Ready to visualize</div>
                            <div>Generate text to see attention patterns</div>
                        </div>
                    </div>"""
                )
        
        # (Visualization output and overlay removed)
        
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
        
        def _generate(prompt, max_tokens, threshold, temperature):
            info, = app.generate_and_visualize(
                prompt, max_tokens, threshold, temperature, "separate"  # Always use separate normalization
            )
            
            # Update visualization and dropdown choices
            max_steps = len(app.current_data['attention_matrices']) - 1 if app.current_data else 0
            viz_html = app.create_d3_visualization_html(step_idx=max_steps, threshold=0.01)  # Start with last step
            token_choices = app.get_token_choices()
            
            return info, viz_html, gr.update(choices=token_choices, value="All tokens"), gr.update(maximum=max_steps, value=max_steps)

        generate_btn.click(
            fn=_generate,
            inputs=[
                prompt_input,
                max_tokens_input,
                gr.State(app.config.DEFAULT_THRESHOLD),  # keep threshold in call but unused
                temperature_input
            ],
            outputs=[generated_info, d3_visualization, token_dropdown, step_slider]
        )
        
        # Event handlers for visualization controls
        def _update_visualization(step_idx, threshold, filter_token="All tokens"):
            """Update visualization when step or threshold changes."""
            viz_html = app.create_d3_visualization_html(step_idx=int(step_idx), threshold=threshold, filter_token=filter_token)
            return viz_html

        def _filter_by_token(selected_token, step_idx, threshold):
            """Update visualization when token filter changes."""
            viz_html = app.create_d3_visualization_html(step_idx=int(step_idx), threshold=threshold, filter_token=selected_token)
            return viz_html

        # Connect visualization controls
        step_slider.change(
            fn=_update_visualization,
            inputs=[step_slider, threshold_slider, token_dropdown],
            outputs=[d3_visualization]
        )
        
        threshold_slider.change(
            fn=_update_visualization,
            inputs=[step_slider, threshold_slider, token_dropdown],
            outputs=[d3_visualization]
        )
        
        token_dropdown.change(
            fn=_filter_by_token,
            inputs=[token_dropdown, step_slider, threshold_slider],
            outputs=[d3_visualization]
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
    """ demo.launch(
        share=False,  # Set to True for public URL
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,  # Default Gradio port
        inbrowser=False  # Don't auto-open browser
    ) """

    demo.launch()