import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .utils import clean_label, scale_weight_to_width, scale_weight_to_opacity

class D3AttentionVisualizer:
    def __init__(self, config):
        self.config = config
        self.current_state = {
            'selected_token': None,
            'selected_type': None,
            'current_step': 0,
            'show_all': True
        }
        
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
        """
        Create D3.js visualization HTML with all data embedded.
        """
        # Clean labels
        input_labels = [clean_label(token) for token in input_tokens]
        output_labels = [clean_label(token) for token in output_tokens]
        
        # Prepare attention data for JavaScript
        attention_data = self._prepare_attention_data(
            attention_matrices, 
            len(input_labels), 
            len(output_labels),
            threshold,
            initial_step,
            selected_token,
            selected_type
        )
        
        # Create HTML with embedded D3 visualization
        header_html = (
            f"""
            <div id="d3-viz-container" style="width: 100%; min-height: {self.config.PLOT_HEIGHT}px;">
                <div style="padding: 10px; background: #f0f0f0; border: 1px solid #ccc; margin-bottom: 10px;">
                    <strong>Debug Info:</strong> {len(input_labels)} input tokens, {len(output_labels)} output tokens, 
                    Step {initial_step}, Threshold {threshold:.3f}
                </div>
                <svg id="attention-viz" width="{self.config.PLOT_WIDTH}" height="{self.config.PLOT_HEIGHT}" style="border: 1px solid #ddd; background: white;">
                    <text x="10" y="30" fill="black">Loading D3 visualization...</text>
                </svg>
            </div>
            """
        )

        config_json = json.dumps({
            'width': self.config.PLOT_WIDTH,
            'height': self.config.PLOT_HEIGHT,
            'nodeSize': self.config.NODE_SIZE,
            'fontSize': self.config.FONT_SIZE,
            'inputColor': self.config.INPUT_COLOR,
            'outputColor': self.config.OUTPUT_COLOR,
            'threshold': threshold,
            'currentStep': initial_step,
            'selectedToken': selected_token,
            'selectedType': selected_type
        })

        script_html = (
            '<script src="https://d3js.org/d3.v7.min.js"></script>\n'
            '<script>\n'
            "console.log('D3 Visualization Script Starting...');\n"
            '\n'
            '// Wait for D3 to load\n'
            'function initVisualization() {\n'
            "    if (typeof d3 === 'undefined') {\n"
            "        console.log('D3 not loaded yet, retrying...');\n"
            '        setTimeout(initVisualization, 100);\n'
            '        return;\n'
            '    }\n'
            '    \n'
            "    console.log('D3 loaded, creating visualization...');\n"
            '    \n'
            '    // Data passed from Python\n'
            '    const inputTokens = ' + json.dumps(input_labels) + ';\n'
            '    const outputTokens = ' + json.dumps(output_labels) + ';\n'
            '    const attentionData = ' + json.dumps(attention_data) + ';\n'
            '    const config = ' + config_json + ';\n'
            '    \n'
            "    console.log('Input tokens:', inputTokens.length);\n"
            "    console.log('Output tokens:', outputTokens.length);\n"
            "    console.log('Attention connections:', attentionData.input_to_output.length);\n"
            '    \n'
            + self._get_d3_visualization_code() +
            '\n'
            '    try {\n'
            '        createAttentionVisualization();\n'
            "        console.log('Visualization created successfully');\n"
            '    } catch (error) {\n'
            "        console.error('Error creating visualization:', error);\n"
            '    }\n'
            '}\n'
            '\n'
            'initVisualization();\n'
            '</script>\n'
        )

        html = header_html + script_html

        return html
    
    def _prepare_attention_data(
        self, 
        attention_matrices: List[Dict],
        num_input: int,
        num_output: int,
        threshold: float,
        current_step: int,
        selected_token: Optional[int] = None,
        selected_type: Optional[str] = None
    ) -> Dict:
        """Prepare attention data for D3 visualization."""
        
        connections = {
            'input_to_output': [],
            'output_to_output': []
        }
        
        # Process input-to-output connections
        for j in range(min(current_step + 1, num_output)):
            if j < len(attention_matrices):
                for i in range(num_input):
                    weight = attention_matrices[j]['input_attention'][i].item()
                    
                    # Apply filtering if token is selected
                    if selected_token is not None and selected_type == 'input':
                        if i != selected_token:
                            continue
                    elif selected_token is not None and selected_type == 'output':
                        if j != selected_token:
                            continue
                    
                    if weight > threshold:
                        connections['input_to_output'].append({
                            'source': i,
                            'target': j,
                            'weight': float(weight),
                            'opacity': scale_weight_to_opacity(weight, threshold),
                            'width': scale_weight_to_width(weight)
                        })
        
        # Process output-to-output connections
        for j in range(1, min(current_step + 1, num_output)):
            if j < len(attention_matrices) and attention_matrices[j]['output_attention'] is not None:
                for i in range(j):
                    if i < len(attention_matrices[j]['output_attention']):
                        weight = attention_matrices[j]['output_attention'][i].item()
                        
                        # Apply filtering if token is selected
                        if selected_token is not None and selected_type == 'output':
                            if i != selected_token and j != selected_token:
                                continue
                        elif selected_token is not None and selected_type == 'input':
                            continue  # Don't show output-to-output when input is selected
                        
                        if weight > threshold:
                            connections['output_to_output'].append({
                                'source': i,
                                'target': j,
                                'weight': float(weight),
                                'opacity': scale_weight_to_opacity(weight, threshold),
                                'width': scale_weight_to_width(weight)
                            })
        
        return connections
    
    def _get_d3_visualization_code(self) -> str:
        """Return the D3.js visualization code."""
        return """
        function createAttentionVisualization() {
            // Clear existing visualization
            d3.select("#attention-viz").selectAll("*").remove();
            
            const svg = d3.select("#attention-viz");
            const width = config.width;
            const height = config.height;
            const margin = {top: 80, right: 200, bottom: 80, left: 150};
            const innerWidth = width - margin.left - margin.right;
            const innerHeight = height - margin.top - margin.bottom;
            
            // Create main group
            const g = svg.append("g")
                .attr("transform", `translate(${margin.left},${margin.top})`);
            
            // Calculate positions
            const inputX = 0;
            const outputX = innerWidth;
            const inputY = d3.scaleLinear()
                .domain([0, inputTokens.length - 1])
                .range([0, innerHeight]);
            const outputY = d3.scaleLinear()
                .domain([0, outputTokens.length - 1])
                .range([0, innerHeight]);
            
            // Draw connections
            const connectionsGroup = g.append("g").attr("class", "connections");
            
            // Input-to-output connections
            attentionData.input_to_output.forEach(conn => {
                connectionsGroup.append("line")
                    .attr("x1", inputX)
                    .attr("y1", inputY(conn.source))
                    .attr("x2", outputX)
                    .attr("y2", outputY(conn.target))
                    .attr("stroke", "blue")
                    .attr("stroke-width", conn.width)
                    .attr("opacity", conn.opacity)
                    .attr("class", "connection input-output")
                    .attr("data-source", conn.source)
                    .attr("data-target", conn.target);
            });
            
            // Output-to-output connections (curved)
            attentionData.output_to_output.forEach(conn => {
                const path = d3.path();
                const y1 = outputY(conn.source);
                const y2 = outputY(conn.target);
                const controlX = outputX + 50;
                
                path.moveTo(outputX, y1);
                path.quadraticCurveTo(controlX, (y1 + y2) / 2, outputX, y2);
                
                connectionsGroup.append("path")
                    .attr("d", path.toString())
                    .attr("fill", "none")
                    .attr("stroke", "orange")
                    .attr("stroke-width", conn.width)
                    .attr("opacity", conn.opacity)
                    .attr("class", "connection output-output")
                    .attr("data-source", conn.source)
                    .attr("data-target", conn.target);
            });
            
            // Draw input nodes
            const inputNodesGroup = g.append("g").attr("class", "input-nodes");
            
            inputTokens.forEach((token, i) => {
                const nodeGroup = inputNodesGroup.append("g")
                    .attr("class", "node-group")
                    .attr("transform", `translate(${inputX}, ${inputY(i)})`);
                
                // Node circle
                nodeGroup.append("circle")
                    .attr("r", config.nodeSize / 2)
                    .attr("fill", config.inputColor)
                    .attr("stroke", "darkblue")
                    .attr("stroke-width", 2)
                    .attr("class", "input-node")
                    .attr("data-index", i)
                    .style("cursor", "pointer")
                    .on("click", function(event) {
                        handleTokenClick(i, 'input');
                    })
                    .on("mouseover", function() {
                        d3.select(this).attr("r", config.nodeSize / 2 + 3);
                    })
                    .on("mouseout", function() {
                        d3.select(this).attr("r", config.nodeSize / 2);
                    });
                
                // Node label
                nodeGroup.append("text")
                    .attr("x", -config.nodeSize / 2 - 10)
                    .attr("y", 5)
                    .attr("text-anchor", "end")
                    .attr("font-size", config.fontSize)
                    .attr("font-family", "Arial, sans-serif")
                    .style("pointer-events", "none")
                    .text(token);
            });
            
            // Draw output nodes
            const outputNodesGroup = g.append("g").attr("class", "output-nodes");
            
            outputTokens.forEach((token, i) => {
                const nodeGroup = outputNodesGroup.append("g")
                    .attr("class", "node-group")
                    .attr("transform", `translate(${outputX}, ${outputY(i)})`);
                
                // Determine node color based on generation step
                const nodeColor = i <= config.currentStep ? config.outputColor : "#e6e6e6";
                
                // Node circle
                nodeGroup.append("circle")
                    .attr("r", config.nodeSize / 2)
                    .attr("fill", nodeColor)
                    .attr("stroke", "darkred")
                    .attr("stroke-width", 2)
                    .attr("class", "output-node")
                    .attr("data-index", i)
                    .style("cursor", "pointer")
                    .on("click", function(event) {
                        handleTokenClick(i, 'output');
                    })
                    .on("mouseover", function() {
                        d3.select(this).attr("r", config.nodeSize / 2 + 3);
                    })
                    .on("mouseout", function() {
                        d3.select(this).attr("r", config.nodeSize / 2);
                    });
                
                // Node label
                nodeGroup.append("text")
                    .attr("x", config.nodeSize / 2 + 10)
                    .attr("y", 5)
                    .attr("text-anchor", "start")
                    .attr("font-size", config.fontSize)
                    .attr("font-family", "Arial, sans-serif")
                    .style("pointer-events", "none")
                    .text(token);
            });
            
            // Add title
            svg.append("text")
                .attr("x", width / 2)
                .attr("y", 30)
                .attr("text-anchor", "middle")
                .attr("font-size", 16)
                .attr("font-weight", "bold")
                .text("Token Attention Flow");
            
            // Add step info
            svg.append("text")
                .attr("x", width / 2)
                .attr("y", height - 20)
                .attr("text-anchor", "middle")
                .attr("font-size", 12)
                .attr("fill", "darkred")
                .text(`Step ${config.currentStep} / ${outputTokens.length - 1}: Generating '${outputTokens[config.currentStep] || ''}'`);
            
            // Highlight selected token if any
            if (config.selectedToken !== null && config.selectedType !== null) {
                const selector = config.selectedType === 'input' ? '.input-node' : '.output-node';
                d3.selectAll(selector)
                    .filter(function() { return +this.getAttribute('data-index') === config.selectedToken; })
                    .attr("fill", "yellow")
                    .attr("r", config.nodeSize / 2 + 4);
            }
        }
        
        function handleTokenClick(index, type) {
            // Update hidden input to trigger Python callback
            const hiddenInput = document.querySelector('#clicked-token-d3 textarea');
            if (hiddenInput) {
                const clickData = JSON.stringify({index: index, type: type});
                hiddenInput.value = clickData;
                hiddenInput.dispatchEvent(new Event('input', { bubbles: true }));
            }
            
            // Update dropdown if it's an input token
            if (type === 'input') {
                const dropdown = document.querySelector('[aria-label="Filter by Input Token"] input');
                if (dropdown && inputTokens[index]) {
                    dropdown.value = inputTokens[index];
                    dropdown.dispatchEvent(new Event('input', { bubbles: true }));
                }
            }
        }
        """