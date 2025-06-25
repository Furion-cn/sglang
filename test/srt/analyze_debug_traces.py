#!/usr/bin/env python3
"""
Debug trace分析脚本
用于对比PyTorch和JAX版本的debug trace文件，分析精度差异
"""
import json
import sys
import os
import argparse
from typing import Dict, List, Any
import numpy as np


def load_trace_file(filepath: str) -> Dict[str, Any]:
    """加载debug trace文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Loaded: {filepath}")
        return data
    except Exception as e:
        print(f"❌ Failed to load {filepath}: {e}")
        return None


def extract_tensor_records(trace_data: Dict[str, Any]) -> Dict[str, List[Dict]]:
    """从trace数据中提取tensor记录"""
    if not trace_data or 'all_forward_records' not in trace_data:
        return {}
    
    return trace_data['all_forward_records']


def compare_tensor_records(pytorch_records: Dict, jax_records: Dict, tolerance: float = 1e-5):
    """对比两个框架的tensor记录"""
    print("\n" + "="*80)
    print("🔍 TENSOR COMPARISON ANALYSIS")
    print("="*80)
    
    # 获取共同的记录键
    pytorch_keys = set(pytorch_records.keys())
    jax_keys = set(jax_records.keys())
    
    common_keys = pytorch_keys & jax_keys
    pytorch_only = pytorch_keys - jax_keys
    jax_only = jax_keys - pytorch_keys
    
    print(f"📊 Record statistics:")
    print(f"   PyTorch records: {len(pytorch_keys)}")
    print(f"   JAX records: {len(jax_keys)}")
    print(f"   Common records: {len(common_keys)}")
    print(f"   PyTorch only: {len(pytorch_only)}")
    print(f"   JAX only: {len(jax_only)}")
    
    if pytorch_only:
        print(f"\n⚠️  Records only in PyTorch: {list(pytorch_only)[:5]}{'...' if len(pytorch_only) > 5 else ''}")
    
    if jax_only:
        print(f"\n⚠️  Records only in JAX: {list(jax_only)[:5]}{'...' if len(jax_only) > 5 else ''}")
    
    # 对比共同的记录
    differences = []
    matches = []
    
    for key in sorted(common_keys):
        pytorch_list = pytorch_records[key]
        jax_list = jax_records[key]
        
        min_len = min(len(pytorch_list), len(jax_list))
        
        for i in range(min_len):
            pt_record = pytorch_list[i]
            jax_record = jax_list[i]
            
            # 对比数值统计
            diff_info = {
                'key': key,
                'step': i,
                'pytorch': pt_record,
                'jax': jax_record,
                'differences': {}
            }
            
            has_difference = False
            
            # 对比形状
            if pt_record.get('shape') != jax_record.get('shape'):
                diff_info['differences']['shape'] = {
                    'pytorch': pt_record.get('shape'),
                    'jax': jax_record.get('shape')
                }
                has_difference = True
            
            # 对比数值统计
            for metric in ['min', 'max', 'mean', 'std']:
                if metric in pt_record and metric in jax_record:
                    pt_val = pt_record[metric]
                    jax_val = jax_record[metric]
                    diff = abs(pt_val - jax_val)
                    
                    if diff > tolerance:
                        diff_info['differences'][metric] = {
                            'pytorch': pt_val,
                            'jax': jax_val,
                            'diff': diff,
                            'relative_diff': diff / max(abs(pt_val), abs(jax_val), 1e-10)
                        }
                        has_difference = True
            
            # 对比布尔标志
            for flag in ['has_nan', 'has_inf']:
                if flag in pt_record and flag in jax_record:
                    if pt_record[flag] != jax_record[flag]:
                        diff_info['differences'][flag] = {
                            'pytorch': pt_record[flag],
                            'jax': jax_record[flag]
                        }
                        has_difference = True
            
            if has_difference:
                differences.append(diff_info)
            else:
                matches.append(diff_info)
    
    print(f"\n📈 Comparison results:")
    print(f"   Matching records: {len(matches)}")
    print(f"   Different records: {len(differences)}")
    print(f"   Tolerance used: {tolerance}")
    
    # 详细报告差异
    if differences:
        print(f"\n❌ FOUND {len(differences)} DIFFERENCES:")
        print("-" * 60)
        
        for i, diff in enumerate(differences[:10]):  # 只显示前10个差异
            print(f"\n{i+1}. Record: {diff['key']} (Step {diff['step']})")
            
            for diff_type, diff_data in diff['differences'].items():
                if diff_type == 'shape':
                    print(f"   📐 Shape: PT={diff_data['pytorch']} vs JAX={diff_data['jax']}")
                elif diff_type in ['has_nan', 'has_inf']:
                    print(f"   🚨 {diff_type}: PT={diff_data['pytorch']} vs JAX={diff_data['jax']}")
                else:
                    pt_val = diff_data['pytorch']
                    jax_val = diff_data['jax']
                    diff_val = diff_data['diff']
                    rel_diff = diff_data['relative_diff']
                    print(f"   📊 {diff_type}: PT={pt_val:.6f} vs JAX={jax_val:.6f} "
                          f"(diff={diff_val:.6f}, rel={rel_diff:.2%})")
        
        if len(differences) > 10:
            print(f"\n... and {len(differences) - 10} more differences")
    
    else:
        print(f"\n✅ ALL RECORDS MATCH within tolerance {tolerance}!")
    
    return len(differences) == 0, differences


def analyze_layer_statistics(pytorch_data: Dict, jax_data: Dict):
    """分析层级统计信息"""
    print("\n" + "="*80)
    print("📊 LAYER STATISTICS ANALYSIS")  
    print("="*80)
    
    pt_summary = pytorch_data.get('summary', {})
    jax_summary = jax_data.get('summary', {})
    
    pt_layer_stats = pt_summary.get('layer_statistics', {})
    jax_layer_stats = jax_summary.get('layer_statistics', {})
    
    print(f"PyTorch layer statistics:")
    print(f"   Total layers: {pt_layer_stats.get('total_layers', 'N/A')}")
    print(f"   Total module types: {pt_layer_stats.get('total_module_types', 'N/A')}")
    print(f"   Total steps: {pt_layer_stats.get('total_steps', 'N/A')}")
    
    print(f"\nJAX layer statistics:")
    print(f"   Total layers: {jax_layer_stats.get('total_layers', 'N/A')}")
    print(f"   Total module types: {jax_layer_stats.get('total_module_types', 'N/A')}")
    print(f"   Total steps: {jax_layer_stats.get('total_steps', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(description='Analyze and compare debug trace files')
    parser.add_argument('pytorch_file', help='PyTorch debug trace JSON file')
    parser.add_argument('jax_file', help='JAX debug trace JSON file') 
    parser.add_argument('--tolerance', type=float, default=1e-5, 
                       help='Tolerance for numerical comparison (default: 1e-5)')
    
    args = parser.parse_args()
    
    print("🔍 Debug Trace Comparison Analysis")
    print("=" * 80)
    
    # 加载文件
    pytorch_data = load_trace_file(args.pytorch_file)
    jax_data = load_trace_file(args.jax_file)
    
    if not pytorch_data or not jax_data:
        print("❌ Failed to load one or both trace files")
        return 1
    
    # 提取tensor记录
    pytorch_records = extract_tensor_records(pytorch_data)
    jax_records = extract_tensor_records(jax_data)
    
    # 对比tensor记录
    all_match, differences = compare_tensor_records(
        pytorch_records, jax_records, args.tolerance)
    
    # 分析层级统计
    analyze_layer_statistics(pytorch_data, jax_data)
    
    # 总结
    print("\n" + "="*80)
    print("🎯 FINAL SUMMARY")
    print("="*80)
    
    if all_match:
        print("✅ ALL TENSORS MATCH - No significant precision differences found!")
    else:
        print(f"⚠️  Found {len(differences)} tensor differences above tolerance {args.tolerance}")
        print("💡 Consider investigating the layers with largest differences")
    
    print(f"\n📁 Files analyzed:")
    print(f"   PyTorch: {args.pytorch_file}")
    print(f"   JAX: {args.jax_file}")
    
    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main()) 