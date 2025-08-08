# Claude Code Instructions - Token Attention Visualizer

## Project Overview
You are helping to build a Token Attention Visualizer - a web-based tool that visualizes attention weights in Large Language Models (LLMs) during text generation. The tool shows how input tokens influence the generation of output tokens through interactive visualizations.

## Core Functionality
1. Accept a text prompt and generate tokens using a Llama model
2. Extract and process attention matrices from the model
3. Create an interactive visualization showing token relationships
4. Allow users to click tokens to filter connections
5. Provide step-by-step navigation through the generation process

## Tech Stack
- **Backend**: FastAPI
- **Frontend**: Gradio (for easy Hugging Face Spaces deployment)
- **Visualization**: Plotly (interactive graphs)
- **ML**: Transformers, PyTorch
- **Models**: Llama models (1B-3B range)

## Project Structure
```
token-attention-viz/
├── app.py                 # Main Gradio application
├── api/
│   ├── __init__.py
│   ├── server.py         # FastAPI endpoints (optional)
│   └── models.py         # Pydantic models
├── core/
│   ├── __init__.py
│   ├── model_handler.py  # Model loading and generation
│   ├── attention.py      # Attention processing
│   └── cache.py          # Caching logic
├── visualization/
│   ├── __init__.py
│   ├── plotly_viz.py     # Plotly visualization
│   └── utils.py          # Token cleaning utilities
├── requirements.txt
└── config.py             # Configuration settings
```

## Implementation Guidelines

### Critical Code to Preserve from Original Implementation

1. **Model Loading Logic**:
   - Device and dtype detection based on GPU capability
   - Pad token handling for models without it
   - Error handling for model loading

2. **Attention Extraction** :
   - BOS token removal from visualization 
   - EOS token handling 
   - Attention matrix extraction with proper indexing

3. **Token Cleaning Function**:
```python
def clean_label(token):
    label = str(token)
    label = label.replace('Ġ', ' ')
    label = label.replace('▁', ' ')
    label = label.replace('Ċ', '\\n')
    label = label.replace('</s>', '[EOS]')
    label = label.replace('<unk>', '[UNK]')
    label = label.replace('<|begin_of_text|>', '[BOS]')
    label = label.replace('<|end_of_text|>', '[EOS]')
    label = re.sub(r'<0x[0-9A-Fa-f]{2}>', '', label)
    return label.strip() if label.strip() else "[EMPTY]"
```

4. **Attention Processing with Separate Normalization**:
   - Layer averaging across heads and layers
   - Separate normalization for input and output attention
   - Epsilon handling (1e-8) to avoid division by zero

5. **Interactive Features**:
   - Token click handling to show specific connections
   - Reset selection functionality
   - Step-by-step navigation
   - "All Connections" view

### Key Implementation Details

#### Model Handler (`core/model_handler.py`)
- Use `unsloth/Llama-3.2-1B-Instruct` as default model
- Implement proper device detection (CUDA if available)
- Use bfloat16 for GPUs with compute capability >= 8.0
- Generate with `output_attentions=True` and `return_dict_in_generate=True`

#### Attention Processing (`core/attention.py`)
- Extract attention for each generation step
- Average across all layers and heads
- Apply separate normalization (input and output attention normalized independently)
- Handle edge cases (first token has no output-to-output attention)

#### Visualization (`visualization/plotly_viz.py`)
- **Layout**:
  - Input tokens on left (x=0.1)
  - Output tokens on right (x=0.9)
  - Use linspace for y-coordinates
- **Connections**:
  - Blue lines for input→output attention
  - Orange curved lines for output→output attention
  - Line thickness proportional to attention weight
  - Only show connections above threshold
- **Interactivity**:
  - Click on any token to filter connections
  - Highlight selected token in yellow
  - Show previously generated tokens in pink
  - Current generating token in coral

#### Gradio Interface (`app.py`)
- **Input Controls**:
  - Text area for prompt
  - Slider for max tokens (1-50)
  - Slider for attention threshold (0.0-0.2, step 0.001)
- **Visualization Controls**:
  - Step slider for navigation
  - Reset Selection button
  - Show All Connections button
- **Display**:
  - Generated text output
  - Interactive Plotly graph

### Performance Optimizations

1. **Caching**:
   - Cache generated attention matrices by prompt+max_tokens hash
   - LRU cache with configurable size (default 10)
   - Store processed attention, not raw tensors

2. **Lazy Updates**:
   - Only update changed traces when stepping through
   - Don't recreate entire plot on threshold change
   - Use Plotly's batch_update for multiple changes

3. **Memory Management**:
   - Clear raw attention tensors after processing
   - Convert to CPU tensors for storage
   - Use float32 instead of original dtype for visualization

### Configuration (`config.py`)
```python
DEFAULT_MODEL = "unsloth/Llama-3.2-1B-Instruct"
DEFAULT_PROMPT = "The old wizard walked through the forest"
DEFAULT_MAX_TOKENS = 20
DEFAULT_THRESHOLD = 0.05
MIN_LINE_WIDTH = 0.5
MAX_LINE_WIDTH = 3.0
PLOT_WIDTH = 1000
PLOT_HEIGHT = 600
```

### Deployment Preparation

For Hugging Face Spaces deployment:
1. Create proper `requirements.txt` with pinned versions
2. Add `README.md` with Spaces metadata
3. Ensure model downloads work in Spaces environment
4. Set appropriate memory/GPU requirements

## Testing Instructions

1. **Basic Functionality**:
   - Test with default prompt
   - Verify attention matrices are extracted correctly
   - Check visualization renders properly

2. **Interactive Features**:
   - Click on input tokens - should show only their connections to outputs
   - Click on output tokens - should show incoming connections
   - Reset button should clear selection
   - Step slider should navigate through generation

3. **Edge Cases**:
   - Empty prompt
   - Single token generation
   - Very long prompts (>100 tokens)
   - High/low threshold values

## Development Workflow

1. Start by implementing the model handler and verify generation works
2. Add attention extraction and processing
3. Create basic visualization without interactivity
4. Add interactive features one by one
5. Implement caching
6. Create Gradio interface
7. Test and optimize performance
8. Prepare for deployment

## Important Notes

- Preserve the token cleaning logic exactly as it handles special tokens
- Keep the BOS token removal logic for cleaner visualization
- Maintain separate normalization (not joint) for attention weights
- Ensure CUDA memory is properly managed to avoid OOM errors
- Test with different model sizes based on available GPU memory

## Common Issues and Solutions

1. **CUDA OOM**: Reduce batch size or use smaller model
2. **Slow Generation**: Enable GPU, use smaller model, or implement streaming
3. **Visualization Lag**: Reduce number of traces, implement virtualization
4. **Cache Misses**: Normalize prompt formatting before hashing

When implementing, prioritize functionality over optimization initially. Get the core visualization working first, then add caching and performance improvements.