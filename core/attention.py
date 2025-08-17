import torch
import numpy as np
from typing import List, Dict, Any, Optional

class AttentionProcessor:
    @staticmethod
    def process_attention_separate(
        attention_data: Dict[str, Any],
        input_tokens: List[str],
        output_tokens: List[str]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Process attention with separate normalization for input and output.
        This preserves the relative importance within each group.
        """
        attentions = attention_data['attentions']
        input_len_for_attention = attention_data['input_len_for_attention']
        output_len = attention_data['output_len']
        
        if not attentions:
            return [{'input_attention': torch.zeros(input_len_for_attention),
                    'output_attention': None} for _ in range(output_len)]
        
        attention_matrices = []
        num_steps = len(attentions)
        
        if num_steps == 0:
            print("Warning: No attention steps found in output.")
            return [{'input_attention': torch.zeros(input_len_for_attention),
                    'output_attention': None} for _ in range(output_len)]
        
        steps_to_process = min(num_steps, output_len)
        
        for i in range(steps_to_process):
            step_attentions = attentions[i]
            input_attention_layers = []
            output_attention_layers = []
            
            for layer_idx, layer_attn in enumerate(step_attentions):
                try:
                    # Extract attention to input tokens (skip BOS token at position 0)
                    input_indices = slice(1, 1 + input_len_for_attention)
                    if layer_attn.shape[3] >= input_indices.stop:
                        # Get attention from current token (position 0 in generation) to input
                        input_attn = layer_attn[0, :, 0, input_indices]
                        input_attention_layers.append(input_attn)
                        
                        # Extract attention to previous output tokens
                        if i > 0:
                            output_indices = slice(1 + input_len_for_attention, 1 + input_len_for_attention + i)
                            if layer_attn.shape[3] >= output_indices.stop:
                                output_attn = layer_attn[0, :, 0, output_indices]
                                output_attention_layers.append(output_attn)
                            else:
                                output_attention_layers.append(
                                    torch.zeros((layer_attn.shape[1], i), device=layer_attn.device)
                                )
                    else:
                        input_attention_layers.append(
                            torch.zeros((layer_attn.shape[1], input_len_for_attention), device=layer_attn.device)
                        )
                        if i > 0:
                            output_attention_layers.append(
                                torch.zeros((layer_attn.shape[1], i), device=layer_attn.device)
                            )
                
                except Exception as e:
                    print(f"Error processing attention at step {i}, layer {layer_idx}: {e}")
                    input_attention_layers.append(
                        torch.zeros((layer_attn.shape[1], input_len_for_attention), device=layer_attn.device)
                    )
                    if i > 0:
                        output_attention_layers.append(
                            torch.zeros((layer_attn.shape[1], i), device=layer_attn.device)
                        )
            
            # Average across layers and heads
            if input_attention_layers:
                avg_input_attn = torch.mean(torch.stack(input_attention_layers).float(), dim=[0, 1])
            else:
                avg_input_attn = torch.zeros(input_len_for_attention)
            
            avg_output_attn = None
            if i > 0 and output_attention_layers:
                avg_output_attn = torch.mean(torch.stack(output_attention_layers).float(), dim=[0, 1])
            elif i > 0:
                avg_output_attn = torch.zeros(i)
            
            # Normalize separately with epsilon for numerical stability
            epsilon = 1e-8
            input_sum = avg_input_attn.sum() + epsilon
            normalized_input_attn = avg_input_attn / input_sum
            
            normalized_output_attn = None
            if i > 0 and avg_output_attn is not None:
                output_sum = avg_output_attn.sum() + epsilon
                normalized_output_attn = avg_output_attn / output_sum
            
            attention_matrices.append({
                'input_attention': normalized_input_attn.cpu(),
                'output_attention': normalized_output_attn.cpu() if normalized_output_attn is not None else None,
                'raw_input_attention': avg_input_attn.cpu(),  # Keep raw for analysis
                'raw_output_attention': avg_output_attn.cpu() if avg_output_attn is not None else None
            })
        
        # Fill remaining steps with zeros if needed
        while len(attention_matrices) < output_len:
            attention_matrices.append({
                'input_attention': torch.zeros(input_len_for_attention),
                'output_attention': None,
                'raw_input_attention': torch.zeros(input_len_for_attention),
                'raw_output_attention': None
            })
        
        return attention_matrices
    
    @staticmethod
    def process_attention_joint(
        attention_data: Dict[str, Any],
        input_tokens: List[str],
        output_tokens: List[str]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Process attention with joint normalization across input and output.
        This preserves the relative importance across all tokens.
        """
        attentions = attention_data['attentions']
        input_len_for_attention = attention_data['input_len_for_attention']
        output_len = attention_data['output_len']
        
        if not attentions:
            return [{'input_attention': torch.zeros(input_len_for_attention),
                    'output_attention': None} for _ in range(output_len)]
        
        attention_matrices = []
        num_steps = len(attentions)
        
        if num_steps == 0:
            print("Warning: No attention steps found in output.")
            return [{'input_attention': torch.zeros(input_len_for_attention),
                    'output_attention': None} for _ in range(output_len)]
        
        steps_to_process = min(num_steps, output_len)
        
        for i in range(steps_to_process):
            step_attentions = attentions[i]
            input_attention_layers = []
            output_attention_layers = []
            
            for layer_idx, layer_attn in enumerate(step_attentions):
                try:
                    # Extract attention to input tokens
                    input_indices = slice(1, 1 + input_len_for_attention)
                    if layer_attn.shape[3] >= input_indices.stop:
                        input_attn = layer_attn[0, :, 0, input_indices]
                        input_attention_layers.append(input_attn)
                        
                        # Extract attention to previous output tokens
                        if i > 0:
                            output_indices = slice(1 + input_len_for_attention, 1 + input_len_for_attention + i)
                            if layer_attn.shape[3] >= output_indices.stop:
                                output_attn = layer_attn[0, :, 0, output_indices]
                                output_attention_layers.append(output_attn)
                            else:
                                output_attention_layers.append(
                                    torch.zeros((layer_attn.shape[1], i), device=layer_attn.device)
                                )
                    else:
                        input_attention_layers.append(
                            torch.zeros((layer_attn.shape[1], input_len_for_attention), device=layer_attn.device)
                        )
                        if i > 0:
                            output_attention_layers.append(
                                torch.zeros((layer_attn.shape[1], i), device=layer_attn.device)
                            )
                
                except Exception as e:
                    print(f"Error processing attention at step {i}, layer {layer_idx}: {e}")
                    input_attention_layers.append(
                        torch.zeros((layer_attn.shape[1], input_len_for_attention), device=layer_attn.device)
                    )
                    if i > 0:
                        output_attention_layers.append(
                            torch.zeros((layer_attn.shape[1], i), device=layer_attn.device)
                        )
            
            # Average across layers and heads
            if input_attention_layers:
                avg_input_attn = torch.mean(torch.stack(input_attention_layers).float(), dim=[0, 1])
            else:
                avg_input_attn = torch.zeros(input_len_for_attention)
            
            avg_output_attn = None
            if i > 0 and output_attention_layers:
                avg_output_attn = torch.mean(torch.stack(output_attention_layers).float(), dim=[0, 1])
            elif i > 0:
                avg_output_attn = torch.zeros(i)
            
            # Joint normalization
            epsilon = 1e-8
            if i > 0 and avg_output_attn is not None:
                # Concatenate and normalize together
                combined_attn = torch.cat([avg_input_attn, avg_output_attn])
                sum_attn = combined_attn.sum() + epsilon
                normalized_combined = combined_attn / sum_attn
                normalized_input_attn = normalized_combined[:input_len_for_attention]
                normalized_output_attn = normalized_combined[input_len_for_attention:]
            else:
                # Only input attention available
                sum_attn = avg_input_attn.sum() + epsilon
                normalized_input_attn = avg_input_attn / sum_attn
                normalized_output_attn = None
            
            attention_matrices.append({
                'input_attention': normalized_input_attn.cpu(),
                'output_attention': normalized_output_attn.cpu() if normalized_output_attn is not None else None
            })
        
        # Fill remaining steps with zeros if needed
        while len(attention_matrices) < output_len:
            attention_matrices.append({
                'input_attention': torch.zeros(input_len_for_attention),
                'output_attention': None
            })
        
        return attention_matrices
    
    @staticmethod
    def extract_attention_for_step(
        attention_data: Dict[str, Any],
        step: int,
        input_len: int
    ) -> Dict[str, torch.Tensor]:
        """
        Extract attention weights for a specific generation step.
        Optimized to only process the needed step.
        """
        attentions = attention_data['attentions']
        
        if step >= len(attentions):
            return {
                'input_attention': torch.zeros(input_len),
                'output_attention': None
            }
        
        step_attentions = attentions[step]
        input_attention_layers = []
        output_attention_layers = []
        
        for layer_attn in step_attentions:
            # Extract input attention
            input_indices = slice(1, 1 + input_len)
            if layer_attn.shape[3] >= input_indices.stop:
                input_attn = layer_attn[0, :, 0, input_indices]
                input_attention_layers.append(input_attn)
                
                # Extract output attention if there are previous outputs
                if step > 0:
                    output_indices = slice(1 + input_len, 1 + input_len + step)
                    if layer_attn.shape[3] >= output_indices.stop:
                        output_attn = layer_attn[0, :, 0, output_indices]
                        output_attention_layers.append(output_attn)
        
        # Average and normalize
        if input_attention_layers:
            avg_input = torch.mean(torch.stack(input_attention_layers).float(), dim=[0, 1])
            normalized_input = avg_input / (avg_input.sum() + 1e-8)
        else:
            normalized_input = torch.zeros(input_len)
        
        normalized_output = None
        if step > 0 and output_attention_layers:
            avg_output = torch.mean(torch.stack(output_attention_layers).float(), dim=[0, 1])
            normalized_output = avg_output / (avg_output.sum() + 1e-8)
        
        return {
            'input_attention': normalized_input.cpu(),
            'output_attention': normalized_output.cpu() if normalized_output is not None else None
        }