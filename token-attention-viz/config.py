from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # Model settings
    DEFAULT_MODEL: str = "unsloth/Llama-3.2-1B-Instruct"
    DEVICE: str = "cuda"  # Will auto-detect
    
    # Generation settings
    DEFAULT_MAX_TOKENS: int = 20
    DEFAULT_PROMPT: str = "The old wizard walked through the forest"
    DEFAULT_TEMPERATURE: float = 0.7
    DEFAULT_TOP_P: float = 0.95
    
    # Visualization settings
    DEFAULT_THRESHOLD: float = 0.05
    MIN_LINE_WIDTH: float = 0.5
    MAX_LINE_WIDTH: float = 3.0
    
    # Colors
    INPUT_COLOR: str = "skyblue"
    OUTPUT_COLOR: str = "coral"
    CONNECTION_COLOR: str = "rgba(128, 128, 128, 0.3)"
    
    # Cache settings
    CACHE_SIZE: int = 10  # Number of generations to cache
    
    # UI settings
    PLOT_WIDTH: int = 1000
    PLOT_HEIGHT: int = 600
    
    # Node settings
    NODE_SIZE: int = 15
    NODE_LINE_WIDTH: float = 2
    
    # Font settings
    FONT_SIZE: int = 10
    FONT_FAMILY: str = "Arial, sans-serif"