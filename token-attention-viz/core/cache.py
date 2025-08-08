from typing import Dict, Any, Optional
import hashlib
import json
import torch
import pickle
import io

class AttentionCache:
    def __init__(self, max_size: int = 10):
        self.cache = {}
        self.access_order = []
        self.max_size = max_size
    
    def get_key(self, prompt: str, max_tokens: int, model: str, temperature: float = 0.7) -> str:
        """Generate cache key from parameters"""
        data = f"{prompt}_{max_tokens}_{model}_{temperature}"
        return hashlib.md5(data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached data"""
        if key in self.cache:
            # Move to end (LRU)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self._deserialize(self.cache[key])
        return None
    
    def set(self, key: str, data: Dict[str, Any]):
        """Store data in cache"""
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = self._serialize(data)
        self.access_order.append(key)
    
    def _serialize(self, data: Dict[str, Any]) -> bytes:
        """Serialize data for caching, handling torch tensors"""
        serialized = {}
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 0:
                # Check if it's a list of dicts with tensors (attention matrices)
                if isinstance(value[0], dict) and any(isinstance(v, torch.Tensor) for v in value[0].values()):
                    # Convert tensors to CPU and serialize
                    serialized_list = []
                    for item in value:
                        serialized_item = {}
                        for k, v in item.items():
                            if isinstance(v, torch.Tensor):
                                serialized_item[k] = v.cpu().numpy()
                            else:
                                serialized_item[k] = v
                        serialized_list.append(serialized_item)
                    serialized[key] = serialized_list
                else:
                    serialized[key] = value
            else:
                serialized[key] = value
        
        buffer = io.BytesIO()
        pickle.dump(serialized, buffer)
        return buffer.getvalue()
    
    def _deserialize(self, data: bytes) -> Dict[str, Any]:
        """Deserialize data from cache, restoring torch tensors"""
        buffer = io.BytesIO(data)
        deserialized = pickle.load(buffer)
        
        # Convert numpy arrays back to tensors where needed
        for key, value in deserialized.items():
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], dict):
                    # Check if it contains numpy arrays (was tensors)
                    import numpy as np
                    for item in value:
                        for k, v in item.items():
                            if isinstance(v, np.ndarray):
                                item[k] = torch.from_numpy(v)
        
        return deserialized
    
    def clear(self):
        """Clear the entire cache"""
        self.cache.clear()
        self.access_order.clear()
    
    def size(self) -> int:
        """Get current cache size"""
        return len(self.cache)