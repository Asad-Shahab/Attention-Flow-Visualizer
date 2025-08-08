import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Optional, List, Dict, Any
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='transformers.generation')

class ModelHandler:
    def __init__(self, model_name: str = None):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_name = model_name
        
    def load_model(self, model_name: str = None) -> Tuple[bool, str]:
        """Load model with optimized settings"""
        if model_name:
            self.model_name = model_name
        
        if not self.model_name:
            return False, "No model name provided"
        
        try:
            print(f"Loading model: {self.model_name}...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Determine device and dtype
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Use bfloat16 for Ampere GPUs (compute capability >= 8.0), otherwise float32
            if self.device == "cuda" and torch.cuda.is_available():
                capability = torch.cuda.get_device_capability()
                if capability[0] >= 8:
                    dtype = torch.bfloat16
                else:
                    dtype = torch.float32
            else:
                dtype = torch.float32
            
            # Load model
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                ).to(self.device)
                print(f"Model loaded on {self.device} with dtype {dtype}")
            except Exception as e:
                print(f"Error loading model with specific dtype: {e}")
                print("Attempting to load without specific dtype...")
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
                print(f"Model loaded on {self.device} (default dtype)")
            
            # Handle pad token
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token:
                    print("Setting pad_token to eos_token")
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    if hasattr(self.model.config, 'pad_token_id') and self.model.config.pad_token_id is None:
                        self.model.config.pad_token_id = self.tokenizer.eos_token_id
                else:
                    print("Warning: No eos_token found to set as pad_token.")
            
            return True, f"Model loaded successfully on {self.device}"
            
        except Exception as e:
            return False, f"Error loading model: {str(e)}"
    
    def generate_with_attention(
        self, 
        prompt: str, 
        max_tokens: int = 30,
        temperature: float = 0.7,
        top_p: float = 0.95
    ) -> Tuple[Optional[List], List[str], List[str], str]:
        """
        Generate text and capture attention weights
        Returns: (attention_matrices, output_tokens, input_tokens, generated_text)
        """
        if not self.model or not self.tokenizer:
            return None, [], [], "Model not loaded"
        
        # Encode input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        input_len_raw = input_ids.shape[1]
        
        print(f"Generating with input length: {input_len_raw}, max_new_tokens: {max_tokens}")
        
        # Generate with attention
        with torch.no_grad():
            attention_mask = torch.ones_like(input_ids)
            gen_kwargs = {
                "attention_mask": attention_mask,
                "max_new_tokens": max_tokens,
                "output_attentions": True,
                "return_dict_in_generate": True,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": temperature > 0
            }
            
            if self.tokenizer.pad_token_id is not None:
                gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
            
            try:
                output = self.model.generate(input_ids, **gen_kwargs)
            except Exception as e:
                print(f"Error during generation: {e}")
                return None, [], [], f"Error during generation: {str(e)}"
        
        # Extract generated tokens
        full_sequence = output.sequences[0]
        if full_sequence.shape[0] > input_len_raw:
            generated_ids = full_sequence[input_len_raw:]
        else:
            generated_ids = torch.tensor([], dtype=torch.long, device=self.device)
        
        # Convert to tokens
        output_tokens = self.tokenizer.convert_ids_to_tokens(generated_ids, skip_special_tokens=False)
        input_tokens_raw = self.tokenizer.convert_ids_to_tokens(input_ids[0], skip_special_tokens=False)
        
        # Handle BOS token removal from visualization
        input_tokens = input_tokens_raw
        input_len_for_attention = input_len_raw
        bos_token = self.tokenizer.bos_token or '<|begin_of_text|>'
        
        if input_tokens_raw and input_tokens_raw[0] == bos_token:
            input_tokens = input_tokens_raw[1:]
            input_len_for_attention = input_len_raw - 1
        
        # Handle EOS token removal
        eos_token = self.tokenizer.eos_token or '<|end_of_text|>'
        if output_tokens and output_tokens[-1] == eos_token:
            output_tokens = output_tokens[:-1]
            generated_ids = generated_ids[:-1]
        
        # Decode generated text
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Extract attention weights
        attentions = getattr(output, 'attentions', None)
        if attentions is None:
            print("Warning: 'attentions' not found in model output. Cannot visualize attention.")
            return None, output_tokens, input_tokens, generated_text
        
        # Return raw attention, tokens, and metadata
        return {
            'attentions': attentions,
            'input_len_for_attention': input_len_for_attention,
            'output_len': len(output_tokens)
        }, output_tokens, input_tokens, generated_text
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.model:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "model_name": self.model_name,
            "device": str(self.device),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "dtype": str(next(self.model.parameters()).dtype),
            "vocab_size": self.tokenizer.vocab_size if self.tokenizer else 0
        }