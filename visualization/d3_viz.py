"""
D3.js visualization module for interactive token attention visualization.
"""

def create_d3_visualization(data):
    """
    Generate a complete, self-contained HTML string with embedded D3.js visualization.
    
    Args:
        data (dict): JSON structure with nodes and links from prepare_d3_data()
        
    Returns:
        str: Complete HTML string with embedded D3.js, CSS, and JavaScript
    """
    
    # Get nodes by type
    input_nodes = [node for node in data.get('nodes', []) if node.get('type') == 'input']
    output_nodes = [node for node in data.get('nodes', []) if node.get('type') == 'output']
    links = data.get('links', [])
    
    # SVG dimensions
    width = 800
    height = max(400, max(len(input_nodes), len(output_nodes)) * 50 + 100)
    
    # Calculate positions
    input_x = 100
    output_x = width - 100
    
    # Position nodes vertically
    def get_y_pos(index, total):
        if total <= 1:
            return height // 2
        return 80 + (index * (height - 160)) / (total - 1)
    
    # Start building SVG
    svg_html = f"""
    <div style='display: flex; flex-direction: column; align-items: center; border: 1px solid #ddd; padding: 20px; margin: 10px; background: white; border-radius: 8px;'>
        <div style='text-align: center; margin-bottom: 15px;'>
            <h3 style='margin: 0; color: #333;'>Token Attention Visualization</h3>
            <p style='margin: 5px 0; color: #666;'>Step {data.get('step', 0) + 1} | {len(input_nodes)} input → {len(output_nodes)} output | {len(links)} connections</p>
        </div>
        
        <svg width="{width}" height="{height}" style='border: 1px solid #eee; background: #fafafa; display: block;'>
            <!-- Background grid -->
            <defs>
                <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
                    <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#f0f0f0" stroke-width="1"/>
                </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#grid)" />
            
            <!-- Column headers -->
            <text x="{input_x}" y="30" text-anchor="middle" font-size="16" font-weight="bold" fill="#4285f4">Input Tokens</text>
            <text x="{output_x}" y="30" text-anchor="middle" font-size="16" font-weight="bold" fill="#ea4335">Output Tokens</text>
    """
    
    # Draw connections first (so they appear behind nodes)
    for link in links:
        # Find source and target nodes
        source_node = next((n for n in input_nodes + output_nodes if n['id'] == link['source']), None)
        target_node = next((n for n in input_nodes + output_nodes if n['id'] == link['target']), None)
        
        if source_node and target_node:
            # Get positions
            if source_node['type'] == 'input':
                source_idx = next((i for i, n in enumerate(input_nodes) if n['id'] == source_node['id']), 0)
                source_y = get_y_pos(source_idx, len(input_nodes))
                source_x_pos = input_x + 20  # Offset from center of node
            else:
                source_idx = next((i for i, n in enumerate(output_nodes) if n['id'] == source_node['id']), 0)
                source_y = get_y_pos(source_idx, len(output_nodes))
                source_x_pos = output_x - 20
            
            if target_node['type'] == 'input':
                target_idx = next((i for i, n in enumerate(input_nodes) if n['id'] == target_node['id']), 0)
                target_y = get_y_pos(target_idx, len(input_nodes))
                target_x_pos = input_x - 20
            else:
                target_idx = next((i for i, n in enumerate(output_nodes) if n['id'] == target_node['id']), 0)
                target_y = get_y_pos(target_idx, len(output_nodes))
                target_x_pos = output_x - 20
            
            # Line properties based on weight
            stroke_width = max(1, min(8, link['weight'] * 20))
            opacity = max(0.3, min(1.0, link['weight'] * 2))
            color = "#4285f4" if link['type'] == 'input_to_output' else "#ea4335"
            
            # Create straight line
            svg_html += f'''
                <line x1="{source_x_pos}" y1="{source_y}" x2="{target_x_pos}" y2="{target_y}" 
                      stroke="{color}" stroke-width="{stroke_width}" opacity="{opacity}"/>
                '''
    
    # Draw input nodes
    for i, node in enumerate(input_nodes):
        y = get_y_pos(i, len(input_nodes))
        token_text = node['token']
        
        # Clean token text - remove special prefix characters
        if token_text.startswith('Ġ'):
            token_text = token_text[1:]  # Remove Ġ prefix
        if token_text.startswith('▁'):
            token_text = token_text[1:]  # Remove ▁ prefix (SentencePiece)
        if token_text.startswith('##'):
            token_text = token_text[2:]  # Remove ## prefix (BERT subwords)
        
        if len(token_text) > 15:
            token_text = token_text[:13] + "..."
        
        svg_html += f'''
            <g>
                <circle cx="{input_x}" cy="{y}" r="12" fill="#4285f4" stroke="#1a73e8" stroke-width="2" opacity="0.9"/>
                <text x="{input_x - 20}" y="{y + 4}" text-anchor="end" font-size="12" fill="#333" font-weight="bold">{token_text}</text>
            </g>
        '''
    
    # Draw output nodes
    for i, node in enumerate(output_nodes):
        y = get_y_pos(i, len(output_nodes))
        token_text = node['token']
        
        # Clean token text - remove special prefix characters
        if token_text.startswith('Ġ'):
            token_text = token_text[1:]  # Remove Ġ prefix
        if token_text.startswith('▁'):
            token_text = token_text[1:]  # Remove ▁ prefix (SentencePiece)
        if token_text.startswith('##'):
            token_text = token_text[2:]  # Remove ## prefix (BERT subwords)
        
        if len(token_text) > 15:
            token_text = token_text[:13] + "..."
        
        svg_html += f'''
            <g>
                <circle cx="{output_x}" cy="{y}" r="12" fill="#ea4335" stroke="#d33b2c" stroke-width="2" opacity="0.9"/>
                <text x="{output_x + 20}" y="{y + 4}" text-anchor="start" font-size="12" fill="#333" font-weight="bold">{token_text}</text>
            </g>
        '''
    
    # Close SVG and add legend
    svg_html += '''
        </svg>
        
        <div style='margin-top: 20px; padding: 16px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;'>
            <div style='display: flex; justify-content: center; align-items: center; gap: 32px; font-size: 12px; color: #64748b; font-family: Inter, sans-serif;'>
                <div style='display: flex; align-items: center; gap: 8px;'>
                    <div style='width: 16px; height: 2px; background: #4285f4; border-radius: 1px;'></div>
                    <span style='color: #1e293b; font-weight: 500;'>Input → Output</span>
                </div>
                <div style='display: flex; align-items: center; gap: 8px;'>
                    <div style='display: flex; gap: 2px;'>
                        <div style='width: 8px; height: 1px; background: #64748b;'></div>
                        <div style='width: 8px; height: 2px; background: #64748b;'></div>
                        <div style='width: 8px; height: 3px; background: #64748b;'></div>
                    </div>
                    <span style='color: #1e293b; font-weight: 500;'>Line thickness = weight</span>
                </div>
            </div>
        </div>
    </div>
    '''
    
    return svg_html

def create_d3_visualization_old(data):
    """
    OLD VERSION - Generate a complete, self-contained HTML string with embedded D3.js visualization.
    
    Args:
        data (dict): JSON structure with nodes and links from prepare_d3_data()
        
    Returns:
        str: Complete HTML string with embedded D3.js, CSS, and JavaScript
    """
    
    html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        .visualization-container {{
            width: 100%;
            height: 600px;
            border: 1px solid #ddd;
            border-radius: 8px;
            background: #fafafa;
            position: relative;
            overflow: hidden;
        }}
        
        .node {{
            cursor: pointer;
            stroke-width: 2px;
        }}
        
        .node.input {{
            fill: #4285f4;
            stroke: #1a73e8;
        }}
        
        .node.output {{
            fill: #ea4335;
            stroke: #d33b2c;
        }}
        
        .node.highlighted {{
            stroke-width: 4px;
            stroke: #ff6d00;
        }}
        
        .node.dimmed {{
            opacity: 0.3;
        }}
        
        .link {{
            stroke: #666;
            stroke-opacity: 0.6;
            fill: none;
        }}
        
        .link.input-to-output {{
            stroke: #4285f4;
        }}
        
        .link.output-to-output {{
            stroke: #ea4335;
        }}
        
        .link.highlighted {{
            stroke-opacity: 1;
            stroke-width: 3px;
        }}
        
        .link.dimmed {{
            stroke-opacity: 0.1;
        }}
        
        .token-label {{
            font-family: 'Courier New', monospace;
            font-size: 12px;
            text-anchor: middle;
            dominant-baseline: central;
            fill: white;
            font-weight: bold;
            pointer-events: none;
        }}
        
        .reset-btn {{
            position: absolute;
            top: 10px;
            right: 10px;
            padding: 8px 16px;
            background: #4285f4;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            z-index: 100;
        }}
        
        .reset-btn:hover {{
            background: #1a73e8;
        }}
        
        .info-panel {{
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: rgba(255, 255, 255, 0.9);
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 11px;
            font-family: Arial, sans-serif;
            border: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="visualization-container" id="viz-container">
        <button class="reset-btn" onclick="resetView()">Reset View</button>
        <div class="info-panel">
            <div>Step: {data.get('step', 0) + 1} / {data.get('total_steps', 1)}</div>
            <div>Nodes: {len(data.get('nodes', []))} | Links: {len(data.get('links', []))}</div>
            <div>Click nodes to filter connections</div>
        </div>
        <svg id="visualization"></svg>
    </div>

    <script>
        // Simple visualization without D3 first - just to test
        const data = {repr(data)};
        
        // Create simple HTML visualization
        const container = document.getElementById("viz-container");
        let html = "<div style='padding: 20px;'>";
        html += "<h3>Debug Info</h3>";
        html += "<p>Nodes: " + data.nodes.length + "</p>";
        html += "<p>Links: " + data.links.length + "</p>";
        
        // Simple SVG without D3
        html += "<svg width='800' height='400' style='border: 1px solid #ccc; background: white;'>";
        
        // Draw input nodes (left side)
        const inputNodes = data.nodes.filter(n => n.type === "input");
        const outputNodes = data.nodes.filter(n => n.type === "output");
        
        inputNodes.forEach((node, i) => {{
            const y = 50 + i * 40;
            html += `<circle cx="50" cy="${{y}}" r="15" fill="#4285f4" stroke="#1a73e8" stroke-width="2"/>`;
            html += `<text x="80" y="${{y + 5}}" font-size="12" fill="black">${{node.token}}</text>`;
        }});
        
        // Draw output nodes (right side)  
        outputNodes.forEach((node, i) => {{
            const y = 50 + i * 40;
            html += `<circle cx="700" cy="${{y}}" r="15" fill="#ea4335" stroke="#d33b2c" stroke-width="2"/>`;
            html += `<text x="620" y="${{y + 5}}" font-size="12" fill="black" text-anchor="end">${{node.token}}</text>`;
        }});
        
        // Draw links
        data.links.forEach(link => {{
            const sourceNode = data.nodes.find(n => n.id === link.source);
            const targetNode = data.nodes.find(n => n.id === link.target);
            if (sourceNode && targetNode) {{
                const sourceIdx = sourceNode.type === "input" ? 
                    inputNodes.findIndex(n => n.id === sourceNode.id) :
                    outputNodes.findIndex(n => n.id === sourceNode.id);
                const targetIdx = targetNode.type === "input" ?
                    inputNodes.findIndex(n => n.id === targetNode.id) :
                    outputNodes.findIndex(n => n.id === targetNode.id);
                    
                const sourceX = sourceNode.type === "input" ? 65 : 685;
                const targetX = targetNode.type === "input" ? 65 : 685;
                const sourceY = 50 + sourceIdx * 40;
                const targetY = 50 + targetIdx * 40;
                
                const strokeWidth = Math.max(1, link.weight * 10);
                const color = link.type === "input_to_output" ? "#4285f4" : "#ea4335";
                
                html += `<line x1="${{sourceX}}" y1="${{sourceY}}" x2="${{targetX}}" y2="${{targetY}}" stroke="${{color}}" stroke-width="${{strokeWidth}}" opacity="0.6"/>`;
            }}
        }});
        
        html += "</svg>";
        html += "</div>";
        
        container.innerHTML = html;
        
    </script>
</body>
</html>
    """
    
    return html_template