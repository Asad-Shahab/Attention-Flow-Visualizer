import plotly.graph_objects as go
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from .utils import (
    clean_label, scale_weight_to_width, scale_weight_to_opacity,
    get_node_positions, create_spline_path, format_attention_text,
    get_color_for_weight, truncate_token_label
)

class AttentionVisualizer:
    def __init__(self, config):
        self.config = config
        self.current_state = {
            'selected_token': None,
            'selected_type': None,
            'current_step': 0,
            'show_all': True
        }
        self.traces_info = {
            'input_to_output': [],
            'output_to_output': [],
            'input_nodes_idx': None,
            'output_nodes_idx': None
        }
        
    def create_interactive_plot(
        self,
        input_tokens: List[str],
        output_tokens: List[str],
        attention_matrices: List[Dict],
        threshold: float = 0.05,
        initial_step: int = 0,
        normalization: str = "separate"
    ) -> go.Figure:
        """
        Create the main interactive visualization.
        """
        # Clean labels
        input_labels = [clean_label(token) for token in input_tokens]
        output_labels = [clean_label(token) for token in output_tokens]
        
        num_input = len(input_labels)
        num_output = len(output_labels)
        num_steps = len(attention_matrices)
        
        if num_input == 0 or num_output == 0 or num_steps == 0:
            return self._create_empty_figure("No data to visualize")
        
        # Get node positions
        input_x, input_y, output_x, output_y = get_node_positions(num_input, num_output)
        
        # Create connection traces
        traces = []
        self.traces_info = {
            'input_to_output': [],
            'output_to_output': [],
            'input_nodes_idx': None,
            'output_nodes_idx': None
        }
        
        # Input to output connections
        for j in range(num_output):
            for i in range(num_input):
                weight = 0
                if j < len(attention_matrices):
                    weight = attention_matrices[j]['input_attention'][i].item()
                
                opacity = scale_weight_to_opacity(weight, threshold=threshold)
                width = scale_weight_to_width(weight) if opacity > 0 else 0.5
                
                trace = go.Scatter(
                    x=[input_x[i], output_x[j]],
                    y=[input_y[i], output_y[j]],
                    mode="lines",
                    line=dict(
                        color=get_color_for_weight(weight, "blue"),
                        width=width
                    ),
                    opacity=opacity,
                    showlegend=False,
                    hoverinfo='text',
                    text=format_attention_text(input_labels[i], output_labels[j], weight),
                    hoverlabel=dict(bgcolor="lightskyblue", bordercolor="darkblue"),
                    name=f"in_to_out_{i}_{j}"
                )
                traces.append(trace)
                self.traces_info['input_to_output'].append({
                    'input_idx': i,
                    'output_idx': j,
                    'trace_idx': len(traces) - 1
                })
        
        # Output to output connections
        for j in range(1, num_output):
            for i in range(j):
                weight = 0
                if j < len(attention_matrices) and attention_matrices[j]['output_attention'] is not None:
                    if i < len(attention_matrices[j]['output_attention']):
                        weight = attention_matrices[j]['output_attention'][i].item()
                
                opacity = scale_weight_to_opacity(weight, threshold=threshold)
                width = scale_weight_to_width(weight) if opacity > 0 else 0.5
                
                # Create spline path for curved connection
                path_x, path_y = create_spline_path(
                    output_x[i], output_y[i],
                    output_x[j], output_y[j],
                    control_offset=0.15
                )
                
                trace = go.Scatter(
                    x=path_x,
                    y=path_y,
                    mode="lines",
                    line=dict(
                        color=get_color_for_weight(weight, "orange"),
                        width=width,
                        shape='spline'
                    ),
                    opacity=opacity,
                    showlegend=False,
                    hoverinfo='text',
                    text=format_attention_text(output_labels[i], output_labels[j], weight),
                    hoverlabel=dict(bgcolor="moccasin", bordercolor="darkorange"),
                    name=f"out_to_out_{i}_{j}"
                )
                traces.append(trace)
                self.traces_info['output_to_output'].append({
                    'from_idx': i,
                    'to_idx': j,
                    'trace_idx': len(traces) - 1
                })
        
        # Input nodes
        input_trace = go.Scatter(
            x=input_x,
            y=input_y,
            mode="markers+text",
            marker=dict(
                size=self.config.NODE_SIZE,
                color=self.config.INPUT_COLOR,
                line=dict(width=self.config.NODE_LINE_WIDTH, color="darkblue")
            ),
            text=[truncate_token_label(label) for label in input_labels],
            textfont=dict(size=self.config.FONT_SIZE, family=self.config.FONT_FAMILY),
            textposition="middle left",
            name="Input Tokens",
            hovertemplate="Input: %{text}<br>Click to filter connections<extra></extra>",
            customdata=[(i, 'input') for i in range(num_input)]
        )
        traces.append(input_trace)
        self.traces_info['input_nodes_idx'] = len(traces) - 1
        
        # Output nodes
        output_colors = []
        for j in range(num_output):
            if j <= initial_step:
                output_colors.append(self.config.OUTPUT_COLOR)
            else:
                output_colors.append("rgba(230, 230, 230, 0.8)")
        
        output_trace = go.Scatter(
            x=output_x,
            y=output_y,
            mode="markers+text",
            marker=dict(
                size=self.config.NODE_SIZE,
                color=output_colors,
                line=dict(width=self.config.NODE_LINE_WIDTH, color="darkred")
            ),
            text=[truncate_token_label(label) for label in output_labels],
            textfont=dict(size=self.config.FONT_SIZE, family=self.config.FONT_FAMILY),
            textposition="middle right",
            name="Output Tokens",
            hovertemplate="Output: %{text}<br>Click to filter connections<extra></extra>",
            customdata=[(i, 'output') for i in range(num_output)]
        )
        traces.append(output_trace)
        self.traces_info['output_nodes_idx'] = len(traces) - 1
        
        # Create figure
        fig = go.Figure(data=traces)
        
        # Update layout
        title = f"Token Attention Flow ({normalization.capitalize()} Normalization)"
        fig.update_layout(
            title=title,
            xaxis=dict(
                range=[-0.1, 1.1],
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                fixedrange=True
            ),
            yaxis=dict(
                range=[0, 1],
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                fixedrange=True
            ),
            hovermode="closest",
            width=self.config.PLOT_WIDTH,
            height=max(self.config.PLOT_HEIGHT, num_input * 30, num_output * 30),
            plot_bgcolor="white",
            margin=dict(l=150, r=150, t=80, b=80),
            hoverdistance=20,
            hoverlabel=dict(font_size=12, font_family=self.config.FONT_FAMILY),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            # Preserve UI state on updates
            uirevision="constant"
        )
        
        # Add legend traces
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color='rgba(0, 0, 255, 0.6)', width=2),
            name='Input→Output'
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color='rgba(255, 165, 0, 0.6)', width=2),
            name='Output→Output'
        ))
        
        # Add annotations
        fig.add_annotation(
            x=0.5, y=0.02,
            text=f"Step {initial_step} / {num_steps-1}: Generating '{output_labels[initial_step] if initial_step < len(output_labels) else ''}'",
            showarrow=False,
            font=dict(size=12, color="darkred"),
            xref="paper", yref="paper"
        )
        
        fig.add_annotation(
            x=0.01, y=0.98,
            text="💡 Click tokens to filter connections | Use step slider to navigate generation",
            showarrow=False,
            font=dict(size=10, color="gray"),
            align="left",
            xref="paper", yref="paper"
        )
        
        self.current_state['current_step'] = initial_step
        
        return fig
    
    def update_for_step(
        self,
        fig: go.Figure,
        step: int,
        attention_matrices: List[Dict],
        output_tokens: List[str],
        threshold: float = 0.05
    ) -> go.Figure:
        """
        Update visualization for a specific generation step.
        """
        if step >= len(attention_matrices):
            return fig
        
        output_labels = [clean_label(token) for token in output_tokens]
        
        with fig.batch_update():
            # Update input-to-output connections for current step
            for conn_info in self.traces_info['input_to_output']:
                if conn_info['output_idx'] == step:
                    weight = attention_matrices[step]['input_attention'][conn_info['input_idx']].item()
                    opacity = scale_weight_to_opacity(weight, threshold=threshold)
                    width = scale_weight_to_width(weight) if opacity > 0 else 0.5
                    
                    trace_idx = conn_info['trace_idx']
                    fig.data[trace_idx].opacity = opacity
                    fig.data[trace_idx].line.width = width
                    fig.data[trace_idx].line.color = get_color_for_weight(weight, "blue")
                elif conn_info['output_idx'] > step:
                    # Hide future connections
                    fig.data[conn_info['trace_idx']].opacity = 0
            
            # Update output-to-output connections
            for conn_info in self.traces_info['output_to_output']:
                if conn_info['to_idx'] == step and attention_matrices[step]['output_attention'] is not None:
                    if conn_info['from_idx'] < len(attention_matrices[step]['output_attention']):
                        weight = attention_matrices[step]['output_attention'][conn_info['from_idx']].item()
                        opacity = scale_weight_to_opacity(weight, threshold=threshold)
                        width = scale_weight_to_width(weight) if opacity > 0 else 0.5
                        
                        trace_idx = conn_info['trace_idx']
                        fig.data[trace_idx].opacity = opacity
                        fig.data[trace_idx].line.width = width
                        fig.data[trace_idx].line.color = get_color_for_weight(weight, "orange")
                elif conn_info['to_idx'] > step:
                    # Hide future connections
                    fig.data[conn_info['trace_idx']].opacity = 0
            
            # Update output node colors
            output_colors = []
            for j in range(len(output_tokens)):
                if j <= step:
                    output_colors.append(self.config.OUTPUT_COLOR)
                else:
                    output_colors.append("rgba(230, 230, 230, 0.8)")
            
            if self.traces_info['output_nodes_idx'] is not None:
                fig.data[self.traces_info['output_nodes_idx']].marker.color = output_colors
            
            # Update step annotation
            fig.layout.annotations[0].text = f"Step {step} / {len(attention_matrices)-1}: Generating '{output_labels[step] if step < len(output_labels) else ''}'"
        
        self.current_state['current_step'] = step
        return fig
    
    def filter_by_token(
        self,
        fig: go.Figure,
        token_idx: int,
        token_type: str,
        attention_matrices: List[Dict],
        threshold: float = 0.05
    ) -> go.Figure:
        """
        Filter connections to show only those related to selected token.
        """
        with fig.batch_update():
            current_step = self.current_state['current_step']
            
            if token_type == 'input':
                # Show only connections from this input token
                for conn_info in self.traces_info['input_to_output']:
                    if conn_info['input_idx'] == token_idx and conn_info['output_idx'] <= current_step:
                        weight = attention_matrices[conn_info['output_idx']]['input_attention'][token_idx].item()
                        opacity = scale_weight_to_opacity(weight, threshold=threshold)
                        fig.data[conn_info['trace_idx']].opacity = opacity if opacity > 0 else 0
                    else:
                        fig.data[conn_info['trace_idx']].opacity = 0
                
                # Hide all output-to-output connections
                for conn_info in self.traces_info['output_to_output']:
                    fig.data[conn_info['trace_idx']].opacity = 0
                    
            elif token_type == 'output':
                # Show connections to this output token
                for conn_info in self.traces_info['input_to_output']:
                    if conn_info['output_idx'] == token_idx and token_idx <= current_step:
                        weight = attention_matrices[token_idx]['input_attention'][conn_info['input_idx']].item()
                        opacity = scale_weight_to_opacity(weight, threshold=threshold)
                        fig.data[conn_info['trace_idx']].opacity = opacity if opacity > 0 else 0
                    else:
                        fig.data[conn_info['trace_idx']].opacity = 0
                
                # Show connections from/to this output token
                for conn_info in self.traces_info['output_to_output']:
                    show = False
                    if conn_info['to_idx'] == token_idx and token_idx <= current_step:
                        if attention_matrices[token_idx]['output_attention'] is not None:
                            if conn_info['from_idx'] < len(attention_matrices[token_idx]['output_attention']):
                                weight = attention_matrices[token_idx]['output_attention'][conn_info['from_idx']].item()
                                opacity = scale_weight_to_opacity(weight, threshold=threshold)
                                fig.data[conn_info['trace_idx']].opacity = opacity if opacity > 0 else 0
                                show = True
                    elif conn_info['from_idx'] == token_idx and conn_info['to_idx'] <= current_step:
                        if attention_matrices[conn_info['to_idx']]['output_attention'] is not None:
                            if token_idx < len(attention_matrices[conn_info['to_idx']]['output_attention']):
                                weight = attention_matrices[conn_info['to_idx']]['output_attention'][token_idx].item()
                                opacity = scale_weight_to_opacity(weight, threshold=threshold)
                                fig.data[conn_info['trace_idx']].opacity = opacity if opacity > 0 else 0
                                show = True
                    
                    if not show:
                        fig.data[conn_info['trace_idx']].opacity = 0
        
        self.current_state['selected_token'] = token_idx
        self.current_state['selected_type'] = token_type
        self.current_state['show_all'] = False
        
        return fig
    
    def show_all_connections(
        self,
        fig: go.Figure,
        attention_matrices: List[Dict],
        threshold: float = 0.05
    ) -> go.Figure:
        """
        Reset to show all connections for current step.
        """
        self.current_state['selected_token'] = None
        self.current_state['selected_type'] = None
        self.current_state['show_all'] = True
        
        return self.update_for_step(
            fig,
            self.current_state['current_step'],
            attention_matrices,
            [clean_label(t) for t in attention_matrices],
            threshold
        )
    
    def _create_empty_figure(self, message: str) -> go.Figure:
        """Create an empty figure with a message."""
        fig = go.Figure()
        fig.update_layout(
            title=message,
            xaxis={'visible': False},
            yaxis={'visible': False},
            width=self.config.PLOT_WIDTH,
            height=self.config.PLOT_HEIGHT
        )
        return fig