import re
from typing import List, Tuple, Optional
import numpy as np

def clean_label(token: str) -> str:
    """
    Cleans token labels for visualization.
    Handles various tokenizer-specific formatting.
    """
    label = str(token)
    
    # Handle common tokenizer prefixes
    label = label.replace('Ġ', ' ')  # GPT-2 style space
    label = label.replace('▁', ' ')  # SentencePiece style space
    label = label.replace('Ċ', '\\n')  # Newline
    
    # Handle special tokens
    label = label.replace('</s>', '[EOS]')
    label = label.replace('<s>', '[BOS]')
    label = label.replace('<unk>', '[UNK]')
    label = label.replace('<pad>', '[PAD]')
    label = label.replace('<|begin_of_text|>', '[BOS]')
    label = label.replace('<|end_of_text|>', '[EOS]')
    label = label.replace('<|endoftext|>', '[EOS]')
    
    # Remove byte-level encoding markers
    label = re.sub(r'<0x[0-9A-Fa-f]{2}>', '', label)
    
    # Clean up whitespace
    label = label.strip()
    
    # Return cleaned label or placeholder
    return label if label else "[EMPTY]"

def scale_weight_to_width(
    weight: float, 
    min_width: float = 0.5, 
    max_width: float = 3.0,
    scale_factor: float = 5.0
) -> float:
    """
    Scale attention weight to line width for visualization.
    
    Args:
        weight: Attention weight (0-1)
        min_width: Minimum line width
        max_width: Maximum line width
        scale_factor: Scaling factor for weight
    
    Returns:
        Scaled line width
    """
    scaled = min(1.0, weight * scale_factor)
    return min_width + (max_width - min_width) * scaled

def scale_weight_to_opacity(
    weight: float,
    min_opacity: float = 0.1,
    max_opacity: float = 1.0,
    threshold: float = 0.0
) -> float:
    """
    Scale attention weight to opacity for visualization.
    
    Args:
        weight: Attention weight (0-1)
        min_opacity: Minimum opacity
        max_opacity: Maximum opacity
        threshold: Threshold below which opacity is 0
    
    Returns:
        Scaled opacity
    """
    if weight < threshold:
        return 0.0
    
    # Linear scaling above threshold
    normalized = (weight - threshold) / (1.0 - threshold) if threshold < 1.0 else weight
    return min_opacity + (max_opacity - min_opacity) * normalized

def get_node_positions(
    num_input: int,
    num_output: int,
    spacing: str = 'linear'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate node positions for visualization.
    
    Args:
        num_input: Number of input tokens
        num_output: Number of output tokens
        spacing: Spacing strategy ('linear', 'equal')
    
    Returns:
        Tuple of (input_x, input_y, output_x, output_y)
    """
    # Y positions (vertical)
    if spacing == 'linear':
        input_y = np.linspace(0.1, 0.9, num_input) if num_input > 1 else np.array([0.5])
        output_y = np.linspace(0.1, 0.9, num_output) if num_output > 1 else np.array([0.5])
    else:  # equal spacing
        total_height = 0.8
        input_spacing = total_height / (num_input + 1)
        output_spacing = total_height / (num_output + 1)
        input_y = np.array([0.1 + (i + 1) * input_spacing for i in range(num_input)])
        output_y = np.array([0.1 + (i + 1) * output_spacing for i in range(num_output)])
    
    # X positions (horizontal)
    input_x = np.full(num_input, 0.1)
    output_x = np.full(num_output, 0.9)
    
    return input_x, input_y, output_x, output_y

def create_spline_path(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    control_offset: float = 0.15
) -> Tuple[List[float], List[float]]:
    """
    Create a spline path for output-to-output connections.
    
    Args:
        start_x, start_y: Starting position
        end_x, end_y: Ending position
        control_offset: Offset for control points
    
    Returns:
        Tuple of (x_path, y_path) for spline
    """
    # Create control points for smooth curve
    path_x = [
        start_x,
        start_x + control_offset,
        end_x + control_offset,
        end_x
    ]
    path_y = [
        start_y,
        start_y,
        end_y,
        end_y
    ]
    
    return path_x, path_y

def format_attention_text(
    from_token: str,
    to_token: str,
    weight: float,
    connection_type: str = "attention"
) -> str:
    """
    Format hover text for attention connections.
    
    Args:
        from_token: Source token
        to_token: Target token
        weight: Attention weight
        connection_type: Type of connection
    
    Returns:
        Formatted hover text
    """
    return (
        f"{from_token} → {to_token}<br>"
        f"{connection_type.capitalize()} Weight: {weight:.4f}"
    )

def get_color_for_weight(
    weight: float,
    base_color: str = "blue",
    use_gradient: bool = True
) -> str:
    """
    Get color for attention weight visualization.
    
    Args:
        weight: Attention weight (0-1)
        base_color: Base color name
        use_gradient: Whether to use gradient based on weight
    
    Returns:
        Color string for plotly
    """
    if not use_gradient:
        if base_color == "blue":
            return "rgba(0, 0, 255, 0.6)"
        elif base_color == "orange":
            return "rgba(255, 165, 0, 0.6)"
        else:
            return "rgba(128, 128, 128, 0.6)"
    
    # Create gradient based on weight
    if base_color == "blue":
        # Light blue to dark blue
        intensity = int(255 - weight * 155)  # 255 to 100
        return f"rgba(0, {intensity}, 255, {0.3 + weight * 0.4})"
    elif base_color == "orange":
        # Light orange to dark orange
        intensity = int(255 - weight * 100)  # 255 to 155
        return f"rgba(255, {intensity}, 0, {0.3 + weight * 0.4})"
    else:
        # Gray scale
        intensity = int(200 - weight * 100)  # 200 to 100
        return f"rgba({intensity}, {intensity}, {intensity}, {0.3 + weight * 0.4})"

def truncate_token_label(token: str, max_length: int = 15) -> str:
    """
    Truncate long token labels for display.
    
    Args:
        token: Token string
        max_length: Maximum length
    
    Returns:
        Truncated token with ellipsis if needed
    """
    cleaned = clean_label(token)
    if len(cleaned) > max_length:
        return cleaned[:max_length-3] + "..."
    return cleaned