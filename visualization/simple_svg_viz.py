import json
from typing import List, Dict, Any, Optional, Tuple
from .utils import clean_label, scale_weight_to_width, scale_weight_to_opacity

class SimpleSVGVisualizer:
    def __init__(self, config):
        self.config = config
        
    def create_visualization_html(
        self,
        input_tokens: List[str],
        output_tokens: List[str],
        attention_matrices: List[Dict],
        threshold: float = 0.05,
        initial_step: int = 0,
        selected_token: Optional[int] = None,
        selected_type: Optional[str] = None
    ) -> str:
        """Create a simple SVG visualization without D3."""
        # Clean labels
        input_labels = [clean_label(token) for token in input_tokens]
        output_labels = [clean_label(token) for token in output_tokens]
        
        # Calculate positions
        width = self.config.PLOT_WIDTH
        height = self.config.PLOT_HEIGHT
        margin = 100
        
        input_x = margin
        output_x = width - margin
        
        # Create SVG elements
        svg_elements = []
        
        # Background
        svg_elements.append(f'<rect width="{width}" height="{height}" fill="white" stroke="#ddd"/>')
        
        # Title
        svg_elements.append(f'<text x="{width/2}" y="30" text-anchor="middle" font-size="16" font-weight="bold">Token Attention Flow</text>')
        
        # Calculate vertical positions
        input_y_positions = []
        output_y_positions = []
        
        if len(input_labels) > 0:
            input_spacing = (height - 2 * margin) / max(1, len(input_labels) - 1)
            input_y_positions = [margin + i * input_spacing for i in range(len(input_labels))]
        
        if len(output_labels) > 0:
            output_spacing = (height - 2 * margin) / max(1, len(output_labels) - 1)
            output_y_positions = [margin + i * output_spacing for i in range(len(output_labels))]
        
        # Draw connections
        for j in range(min(initial_step + 1, len(output_labels))):
            if j < len(attention_matrices):
                for i in range(len(input_labels)):
                    weight = attention_matrices[j]['input_attention'][i].item()
                    
                    # Apply filtering
                    if selected_token is not None:
                        if selected_type == 'input' and i != selected_token:
                            continue
                        elif selected_type == 'output' and j != selected_token:
                            continue
                    
                    if weight > threshold:
                        opacity = scale_weight_to_opacity(weight, threshold)
                        width_val = scale_weight_to_width(weight)
                        
                        svg_elements.append(
                            f'<line x1="{input_x}" y1="{input_y_positions[i]}" '
                            f'x2="{output_x}" y2="{output_y_positions[j]}" '
                            f'stroke="blue" stroke-width="{width_val}" opacity="{opacity}"/>'
                        )
        
        # Draw input nodes
        for i, label in enumerate(input_labels):
            y = input_y_positions[i]
            color = "yellow" if selected_token == i and selected_type == 'input' else self.config.INPUT_COLOR
            
            svg_elements.append(
                f'<circle cx="{input_x}" cy="{y}" r="{self.config.NODE_SIZE/2}" '
                f'fill="{color}" stroke="darkblue" stroke-width="2" '
                f'style="cursor: pointer" '
                f'onclick="handleTokenClick({i}, \'input\')"/>'
            )
            svg_elements.append(
                f'<text x="{input_x - self.config.NODE_SIZE/2 - 10}" y="{y + 5}" '
                f'text-anchor="end" font-size="{self.config.FONT_SIZE}">{label}</text>'
            )
        
        # Draw output nodes
        for j, label in enumerate(output_labels):
            y = output_y_positions[j]
            color = "yellow" if selected_token == j and selected_type == 'output' else (
                self.config.OUTPUT_COLOR if j <= initial_step else "#e6e6e6"
            )
            
            svg_elements.append(
                f'<circle cx="{output_x}" cy="{y}" r="{self.config.NODE_SIZE/2}" '
                f'fill="{color}" stroke="darkred" stroke-width="2" '
                f'style="cursor: pointer" '
                f'onclick="handleTokenClick({j}, \'output\')"/>'
            )
            svg_elements.append(
                f'<text x="{output_x + self.config.NODE_SIZE/2 + 10}" y="{y + 5}" '
                f'text-anchor="start" font-size="{self.config.FONT_SIZE}">{label}</text>'
            )
        
        # Step info
        svg_elements.append(
            f'<text x="{width/2}" y="{height - 20}" text-anchor="middle" font-size="12" fill="darkred">'
            f'Step {initial_step} / {len(output_labels) - 1}: Generating "{output_labels[initial_step] if initial_step < len(output_labels) else ""}"'
            f'</text>'
        )
        
        # Create HTML
        html = f"""
        <div style="width: 100%; overflow-x: auto;">
            <svg width="{width}" height="{height}" style="border: 1px solid #ddd;">
                {''.join(svg_elements)}
            </svg>
        </div>
        
        <script>
        function handleTokenClick(index, type) {{
            console.log('Token clicked:', index, type);
            const hiddenInput = document.querySelector('#clicked-token-d3 textarea');
            if (hiddenInput) {{
                const clickData = JSON.stringify({{index: index, type: type}});
                hiddenInput.value = clickData;
                hiddenInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
            }}
        }}
        </script>
        """
        
        return html