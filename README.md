---
title: Attention Flow Visualizer
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: apache-2.0
tags:
- attention
- visualization
- transformers
- nlp
- machine-learning
---

# Attention Flow Visualizer

An interactive tool for visualizing attention patterns in Large Language Models during text generation.

## Features

- 🚀 **Real-time Generation**: Generate text with any Hugging Face model
- 🔍 **Attention Visualization**: Explore attention patterns with clear visual representations
- ⚡ **Smart Caching**: Fast response with intelligent result caching
- 🎯 **Token Selection**: Use dropdown menus to select and filter token connections
- 📈 **Step Navigation**: Navigate through generation steps
- 🎨 **Customizable Threshold**: Filter weak attention connections

## How It Works

The visualizer shows how tokens attend to each other during text generation:
- **Blue lines**: Attention from input tokens to output tokens
- **Line thickness**: Represents attention weight strength

## Usage

1. **Load a Model**: Enter a Hugging Face model name (default: HuggingFaceTB/SmolLM-135M-Instruct)
2. **Enter Prompt**: Type your input text
3. **Configure Settings**: Adjust max tokens, temperature, and normalization
4. **Generate**: Click to generate text and visualize attention
5. **Explore**: Use dropdown menus to select tokens and view their attention patterns

## Technical Details

- Built with Gradio for the interface
- Visualization system with dropdown-based token selection
- Supports any Hugging Face causal language model
- Optimized for smaller models like SmolLM for efficient deployment
- Implements efficient attention processing and caching

## Local Development

```bash
# Clone the repository
git clone https://github.com/Asad-Shahab/attention-flow-visualizer
cd attention-flow-visualizer

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

## Requirements

- Python 3.8+
- 4GB+ RAM (SmolLM models are lightweight)
- GPU acceleration optional (works well on CPU)

