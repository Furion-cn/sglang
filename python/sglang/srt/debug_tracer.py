import threading
from typing import Dict, List, Optional, Union, Any

import torch
import jax.numpy as jnp


class UnifiedDebugTracer:    
    def __init__(self):
        self.enabled = True
        self.records = {}
        self.lock = threading.Lock() 
    
    def print(self, tensor: Union[torch.Tensor, jnp.ndarray], name: str, stage: str = "", extra_info: str = ""):
        if not self.enabled:
            return
        
        if tensor is None:
            print(f"[{stage}] {name}: None")
            return
        
        key = f"{stage}_{name}" if stage else name
        
        if hasattr(tensor, 'cpu'): 
            stats = self._compute_pytorch_stats(tensor, name, stage)
        else:  # JAX
            stats = self._compute_jax_stats(tensor, name, stage, extra_info)
        
        with self.lock:
            if key not in self.records:
                self.records[key] = []
            self.records[key].append(stats)
        
        # 打印统计信息
        self._print_stats(stats, key)
    
    def _compute_pytorch_stats(self, tensor: torch.Tensor, name: str, stage: str) -> Dict[str, Any]:
        if hasattr(tensor, 'cpu'):
            tensor_cpu = tensor.cpu()
        else:
            tensor_cpu = tensor
        
        try:
            stats = {
                'framework': 'pytorch',
                'name': name,
                'stage': stage,
                'shape': tuple(tensor.shape),
                'dtype': str(tensor.dtype),
                'min': float(tensor_cpu.min()),
                'max': float(tensor_cpu.max()),
                'mean': float(tensor_cpu.mean()),
                'std': float(tensor_cpu.std()),
                'has_nan': torch.isnan(tensor_cpu).any().item(),
                'has_inf': torch.isinf(tensor_cpu).any().item(),
            }
        except Exception as e:
            stats = {
                'framework': 'pytorch',
                'name': name,
                'stage': stage,
                'shape': tuple(tensor.shape),
                'dtype': str(tensor.dtype),
                'error': str(e)
            }
        
        return stats
    
    def _compute_jax_stats(self, tensor: jnp.ndarray, name: str, stage: str, extra_info: str) -> Dict[str, Any]:
        try:
            stats = {
                'framework': 'jax',
                'name': name,
                'stage': stage,
                'shape': tuple(tensor.shape),
                'dtype': str(tensor.dtype),
                'min': float(jnp.min(tensor).item()),
                'max': float(jnp.max(tensor).item()),
                'mean': float(jnp.mean(tensor).item()),
                'std': float(jnp.std(tensor).item()),
                'has_nan': bool(jnp.any(jnp.isnan(tensor)).item()),
                'has_inf': bool(jnp.any(jnp.isinf(tensor)).item()),
                'extra_info': extra_info
            }
        except Exception as e:
            stats = {
                'framework': 'jax',
                'name': name,
                'stage': stage,
                'shape': tuple(tensor.shape),
                'dtype': str(tensor.dtype),
                'extra_info': extra_info,
                'error': str(e)
            }
        
        return stats
    
    def _print_stats(self, stats: Dict[str, Any], key: str):
        if 'error' in stats:
            print(f"[{stats['stage']}] {stats['name']}: shape={stats['shape']}, dtype={stats['dtype']}, error={stats['error']}")
        else:
            framework = stats['framework'].upper()
            extra = f" {stats.get('extra_info', '')}" if stats.get('extra_info') else ""
            nan_inf = ""
            if stats['has_nan']:
                nan_inf += ", HAS_NAN"
            if stats['has_inf']:
                nan_inf += ", HAS_INF"
            
            print(f"[{framework}][{stats['stage']}] {stats['name']}: shape={stats['shape']}, "
                  f"min={stats['min']:.6f}, max={stats['max']:.6f}, "
                  f"mean={stats['mean']:.6f}, std={stats['std']:.6f}{nan_inf}{extra}")
    
    def get_records(self, key: str = None) -> Union[Dict[str, List], List]:
        with self.lock:
            if key is None:
                return dict(self.records)
            return self.records.get(key, [])
    
    def clear_records(self):
        with self.lock:
            self.records.clear()
    
    def enable(self):
        self.enabled = True
    
    def disable(self):
        self.enabled = False
    
    def compare_frameworks(self, jax_key: str, pytorch_key: str, tolerance: float = 1e-5) -> bool:
        jax_records = self.get_records(jax_key)
        pytorch_records = self.get_records(pytorch_key)
        
        if not jax_records or not pytorch_records:
            print(f"Warning: 记录 {jax_key} 或 {pytorch_key} 为空")
            return False
        
        print(f"\n=== 比较 {jax_key} (JAX) vs {pytorch_key} (PyTorch) ===")
        min_len = min(len(jax_records), len(pytorch_records))
        
        all_match = True
        for i in range(min_len):
            jax_record = jax_records[i]
            pytorch_record = pytorch_records[i]
            
            print(f"Step {i}:")
            print(f"  Shape: {jax_record['shape']} vs {pytorch_record['shape']}")
            
            if jax_record['shape'] != pytorch_record['shape']:
                print(f"  ❌ Shape mismatch!")
                all_match = False
                continue
            
            for metric in ['min', 'max', 'mean', 'std']:
                if metric in jax_record and metric in pytorch_record:
                    diff = abs(jax_record[metric] - pytorch_record[metric])
                    match = diff <= tolerance
                    status = "✅" if match else "❌"
                    print(f"  {metric.capitalize()}: {jax_record[metric]:.6f} vs {pytorch_record[metric]:.6f} "
                          f"(diff: {diff:.6f}) {status}")
                    if not match:
                        all_match = False
            
            for flag in ['has_nan', 'has_inf']:
                if flag in jax_record and flag in pytorch_record:
                    match = jax_record[flag] == pytorch_record[flag]
                    status = "✅" if match else "❌"
                    print(f"  {flag}: {jax_record[flag]} vs {pytorch_record[flag]} {status}")
                    if not match:
                        all_match = False
            
            print()
        
        return all_match
    
    def compare_records(self, other_records: List[Dict], tolerance: float = 1e-5) -> bool:
        all_records = []
        for records_list in self.records.values():
            all_records.extend(records_list)
        
        if len(all_records) != len(other_records):
            print(f"Record count mismatch: {len(all_records)} vs {len(other_records)}")
            return False
        
        all_match = True
        for i, (record1, record2) in enumerate(zip(all_records, other_records)):
            if record1['name'] != record2['name'] or record1['stage'] != record2['stage']:
                print(f"Record {i}: name/stage mismatch")
                all_match = False
                continue
            
            for key in ['min', 'max', 'mean', 'std']:
                if key in record1 and key in record2:
                    diff = abs(record1[key] - record2[key])
                    if diff > tolerance:
                        print(f"Record {i} ({record1['name']}): {key} differs by {diff:.8f}")
                        all_match = False
            
            for key in ['has_nan', 'has_inf']:
                if key in record1 and key in record2:
                    if record1[key] != record2[key]:
                        print(f"Record {i} ({record1['name']}): {key} differs")
                        all_match = False
        
        return all_match


global_tracer = UnifiedDebugTracer()