#!/usr/bin/env python3
"""
PyTorch vs JAX Debug Trace 精度比较工具
支持按forward_step进行详细对比分析
"""
import json
import sys
import os
import argparse
from typing import Dict, List, Any, Tuple
import numpy as np
from collections import defaultdict


def load_trace_file(filepath: str) -> Dict[str, Any]:
    """加载debug trace文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ 成功加载: {os.path.basename(filepath)}")
        
        # 验证数据结构
        if 'all_forward_records' not in data:
            print(f"⚠️  警告: {filepath} 没有 'all_forward_records' 字段")
            return None
            
        records_count = len(data['all_forward_records'])
        print(f"   包含 {records_count} 种记录类型")
        return data
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析错误 {filepath}: 第{e.lineno}行第{e.colno}列 - {e.msg}")
        return None
    except FileNotFoundError:
        print(f"❌ 文件不存在: {filepath}")
        return None
    except Exception as e:
        print(f"❌ 加载失败 {filepath}: {e}")
        return None


def extract_records_by_step(trace_data: Dict[str, Any]) -> Dict[int, Dict[str, List[Dict]]]:
    """按forward_step组织记录"""
    if not trace_data or 'all_forward_records' not in trace_data:
        return {}
    
    all_records = trace_data['all_forward_records']
    step_organized = defaultdict(lambda: defaultdict(list))
    
    # 创建记录键的标准化映射
    normalized_keys = {}
    for record_key in all_records.keys():
        # 将JAX中的___call___转换为_forward_
        normalized_key = record_key.replace('___call___', '_forward_')
        normalized_keys[record_key] = normalized_key
    
    for record_key, record_list in all_records.items():
        normalized_key = normalized_keys[record_key]
        for record in record_list:
            step = record.get('forward_step', 0)
            step_organized[step][normalized_key].append(record)
    
    return dict(step_organized)


def compare_single_record(pt_record: Dict, jax_record: Dict, tolerance: float = 1e-5) -> Tuple[bool, Dict]:
    """比较单个记录"""
    differences = {}
    has_difference = False
    
    # 比较形状
    pt_shape = pt_record.get('shape', [])
    jax_shape = jax_record.get('shape', [])
    if pt_shape != jax_shape:
        differences['shape'] = {
            'pytorch': pt_shape,
            'jax': jax_shape
        }
        has_difference = True
    
    # 比较数值统计
    for metric in ['min', 'max', 'mean', 'std']:
        if metric in pt_record and metric in jax_record:
            pt_val = pt_record[metric]
            jax_val = jax_record[metric]
            
            # 处理NaN情况
            if np.isnan(pt_val) or np.isnan(jax_val):
                if not (np.isnan(pt_val) and np.isnan(jax_val)):
                    differences[metric] = {
                        'pytorch': pt_val,
                        'jax': jax_val,
                        'diff': 'NaN_mismatch',
                        'relative_diff': float('inf')
                    }
                    has_difference = True
                continue
            
            diff = abs(pt_val - jax_val)
            if diff > tolerance:
                rel_diff = diff / max(abs(pt_val), abs(jax_val), 1e-10)
                differences[metric] = {
                    'pytorch': pt_val,
                    'jax': jax_val,
                    'diff': diff,
                    'relative_diff': rel_diff
                }
                has_difference = True
    
    # 比较布尔标志
    for flag in ['has_nan', 'has_inf']:
        if flag in pt_record and flag in jax_record:
            if pt_record[flag] != jax_record[flag]:
                differences[flag] = {
                    'pytorch': pt_record[flag],
                    'jax': jax_record[flag]
                }
                has_difference = True
    
    return not has_difference, differences


def compare_step_records(pt_step_data: Dict, jax_step_data: Dict, step: int, tolerance: float) -> Dict:
    """比较特定step的所有记录"""
    step_result = {
        'step': step,
        'total_comparisons': 0,
        'matches': 0,
        'match_details': [],
        'differences': 0,
        'details': [],
    }
    
    # 获取共同的记录键
    pt_keys = set(pt_step_data.keys())
    jax_keys = set(jax_step_data.keys())
    common_keys = pt_keys & jax_keys
    
    for key in sorted(common_keys):
        pt_records = pt_step_data[key]
        jax_records = jax_step_data[key]
        
        # 通常每个key在每个step只有一个记录，但也可能有多个
        min_len = min(len(pt_records), len(jax_records))
        
        for i in range(min_len):
            step_result['total_comparisons'] += 1
            
            is_match, differences = compare_single_record(
                pt_records[i], jax_records[i], tolerance)
            
            if is_match:
                step_result['matches'] += 1
                step_result['match_details'].append({
                    'record_key': key,
                    'record_index': i,
                    'differences': differences,
                    'pytorch_record': pt_records[i],
                    'jax_record': jax_records[i]
                })
            else:
                step_result['differences'] += 1
                step_result['details'].append({
                    'record_key': key,
                    'record_index': i,
                    'differences': differences,
                    'pytorch_record': pt_records[i],
                    'jax_record': jax_records[i]
                })
    
    return step_result


def identify_key_components(record_key: str) -> Dict[str, str]:
    """识别记录的关键组件信息"""
    key_lower = record_key.lower()
    
    component_info = {
        'type': 'unknown',
        'layer': 'unknown',
        'stage': 'unknown'
    }
    
    # 识别组件类型
    if 'embedding' in key_lower:
        component_info['type'] = 'embedding'
        component_info['layer'] = 'all'
    elif 'rmsnorm' in key_lower:
        component_info['type'] = 'rmsnorm'
        if 'final' in key_lower:
            component_info['layer'] = 'final'
        elif 'layer_id_' in key_lower:
            try:
                layer_id = key_lower.split('layer_id_')[1].split('_')[0]
                component_info['layer'] = layer_id
            except:
                pass
    elif 'attention' in key_lower:
        component_info['type'] = 'attention'
        if 'layer_id_' in key_lower:
            try:
                layer_id = key_lower.split('layer_id_')[1].split('_')[0]
                component_info['layer'] = layer_id
            except:
                pass
    elif 'mlp' in key_lower:
        component_info['type'] = 'mlp'
        if 'layer_id_' in key_lower:
            try:
                layer_id = key_lower.split('layer_id_')[1].split('_')[0]
                component_info['layer'] = layer_id
            except:
                pass
    elif 'decoder_layer' in key_lower:
        component_info['type'] = 'decoder_layer'
        if 'layer_id_' in key_lower:
            try:
                layer_id = key_lower.split('layer_id_')[1].split('_')[0]
                component_info['layer'] = layer_id
            except:
                pass

    # 识别阶段
    if 'input' in key_lower:
        component_info['stage'] = 'input'
    elif 'output' in key_lower:
        component_info['stage'] = 'output'
    
    return component_info


def analyze_difference_patterns(all_step_results: List[Dict]) -> Dict:
    """分析差异模式"""
    patterns = {
        'by_component': defaultdict(int),
        'by_layer': defaultdict(int),
        'by_step': defaultdict(int),
        'by_metric': defaultdict(int),
        'worst_differences': []
    }
    
    for step_result in all_step_results:
        step = step_result['step']
        patterns['by_step'][step] = step_result['differences']
        
        for detail in step_result['details']:
            record_key = detail['record_key']
            component_info = identify_key_components(record_key)
            
            patterns['by_component'][component_info['type']] += 1
            patterns['by_layer'][component_info['layer']] += 1
            
            # 记录最大差异
            for metric, diff_data in detail['differences'].items():
                if metric in ['min', 'max', 'mean', 'std'] and isinstance(diff_data, dict):
                    if 'relative_diff' in diff_data:
                        patterns['by_metric'][metric] += 1
                        patterns['worst_differences'].append({
                            'step': step,
                            'record_key': record_key,
                            'metric': metric,
                            'relative_diff': diff_data['relative_diff'],
                            'absolute_diff': diff_data['diff'],
                            'pytorch_val': diff_data['pytorch'],
                            'jax_val': diff_data['jax']
                        })
    
    # 排序最严重的差异
    patterns['worst_differences'].sort(
        key=lambda x: x['relative_diff'] if x['relative_diff'] != float('inf') else 1e10, 
        reverse=True)
    
    return patterns


def print_step_summary(step_results: List[Dict]):
    """打印按步骤的总结"""
    print("\n" + "="*80)
    print("📊 按步骤对比总结")
    print("="*80)
    
    for result in step_results:
        step = result['step']
        total = result['total_comparisons']
        matches = result['matches']
        diffs = result['differences']
        
        if total > 0:
            match_rate = (matches / total) * 100
            status = "✅" if diffs == 0 else "⚠️" if diffs < total * 0.1 else "❌"
            
            print(f"{status} Step {step}: {matches}/{total} 匹配 ({match_rate:.1f}%), {diffs} 个差异")
            
            # 显示该步骤的主要差异
            if diffs > 0 and len(result['details']) > 0:
                print(f"     主要差异:")
                for i, detail in enumerate(result['details'][:3]):  # 只显示前3个
                    key = detail['record_key']
                    component = identify_key_components(key)
                    print(f"       {i+1}. {component['type']} (layer {component['layer']}) - {component['stage']}")
                if len(result['details']) > 3:
                    print(f"       ... 还有 {len(result['details']) - 3} 个差异")

def print_match_details(step_results: List[Dict]):
    """打印匹配详情"""
    print("\n" + "="*80)
    print("🔍 匹配详情")
    print("="*80)
    
    for result in step_results:
        if result['matches'] > 0:
            print(f"Step {result['step']}: {result['matches']} 个匹配")
            for detail in result['match_details']:
                print(f"   {detail['record_key']}")


def print_pattern_analysis(patterns: Dict):
    """打印模式分析"""
    print("\n" + "="*80)
    print("🔍 差异模式分析")
    print("="*80)
    
    print("📈 按组件类型:")
    for component, count in sorted(patterns['by_component'].items()):
        print(f"   {component}: {count} 个差异")
    
    print("\n📈 按层级:")
    for layer, count in sorted(patterns['by_layer'].items()):
        print(f"   Layer {layer}: {count} 个差异")
    
    print("\n📈 按指标:")
    for metric, count in sorted(patterns['by_metric'].items()):
        print(f"   {metric}: {count} 个差异")
    
    print("\n🔥 最严重的差异 (Top 5):")
    for i, diff in enumerate(patterns['worst_differences'][:5]):
        step = diff['step']
        key = diff['record_key']
        metric = diff['metric']
        rel_diff = diff['relative_diff']
        abs_diff = diff['absolute_diff']
        
        if rel_diff == float('inf'):
            print(f"   {i+1}. Step {step} - {key} - {metric}: 无穷大差异")
        else:
            print(f"   {i+1}. Step {step} - {key} - {metric}: {rel_diff:.2%} (绝对差异: {abs_diff:.2e})")


def print_detailed_differences(step_results: List[Dict], max_details: int = 10):
    """打印详细差异信息"""
    print("\n" + "="*80)
    print(f"🔍 详细差异分析 (显示前 {max_details} 个)")
    print("="*80)
    
    # 收集所有差异并按层级排序
    all_differences = []
    for step_result in step_results:
        step = step_result['step']
        for detail in step_result['details']:
            record_key = detail['record_key']
            component_info = identify_key_components(record_key)
            
            # 提取层级并转换为整数以便正确排序
            try:
                layer_id = int(component_info['layer']) if component_info['layer'].isdigit() else float('inf')
            except:
                layer_id = float('inf')  # 对于无法解析的层级，放到最后
                
            all_differences.append({
                'step': step,
                'record_key': record_key,
                'component_info': component_info,
                'layer_id': layer_id,
                'differences': detail['differences'],
                'detail': detail
            })
    
    # 按步骤和层级排序
    all_differences.sort(key=lambda x: (x['step'], x['layer_id']))
    
    # 显示排序后的差异
    for idx, diff in enumerate(all_differences[:max_details]):
        if idx >= max_details:
            break
            
        step = diff['step']
        record_key = diff['record_key']
        component_info = diff['component_info']
        
        print(f"\n{idx + 1}. Step {step} - {record_key}")
        print(f"    组件: {component_info['type']} (Layer {component_info['layer']}) - {component_info['stage']}")
        
        for diff_type, diff_data in diff['differences'].items():
            if diff_type == 'shape':
                print(f"    📐 Shape差异: PT={diff_data['pytorch']} vs JAX={diff_data['jax']}")
            elif diff_type in ['has_nan', 'has_inf']:
                print(f"    🚨 {diff_type}: PT={diff_data['pytorch']} vs JAX={diff_data['jax']}")
            elif isinstance(diff_data, dict) and 'relative_diff' in diff_data:
                pt_val = diff_data['pytorch']
                jax_val = diff_data['jax']
                rel_diff = diff_data['relative_diff']
                abs_diff = diff_data['diff']
                print(f"    📊 {diff_type}: PT={pt_val:.6f} vs JAX={jax_val:.6f} "
                      f"(相对差异: {rel_diff:.2%}, 绝对差异: {abs_diff:.2e})")


def generate_conclusion(step_results: List[Dict], patterns: Dict, tolerance: float) -> str:
    """生成结论"""
    total_comparisons = sum(r['total_comparisons'] for r in step_results)
    total_matches = sum(r['matches'] for r in step_results)
    total_differences = sum(r['differences'] for r in step_results)
    
    if total_comparisons == 0:
        return "❌ 无法进行比较，没有找到匹配的记录"
    
    match_rate = (total_matches / total_comparisons) * 100
    
    conclusion_parts = []
    
    # 总体评估
    if total_differences == 0:
        conclusion_parts.append("✅ **完美匹配**: PyTorch和JAX版本在所有测量指标上完全一致")
    elif match_rate >= 95:
        conclusion_parts.append(f"✅ **高度一致**: {match_rate:.1f}% 的指标匹配，少量差异在可接受范围内")
    elif match_rate >= 80:
        conclusion_parts.append(f"⚠️ **基本一致**: {match_rate:.1f}% 的指标匹配，存在一些需要关注的差异")
    else:
        conclusion_parts.append(f"❌ **存在明显差异**: 仅 {match_rate:.1f}% 的指标匹配，需要深入调查")
    
    # 差异分布分析
    if total_differences > 0:
        most_affected_component = max(patterns['by_component'].items(), key=lambda x: x[1])
        most_affected_layer = max(patterns['by_layer'].items(), key=lambda x: x[1])
        
        conclusion_parts.append(f"🔍 **差异集中在**: {most_affected_component[0]} 组件 "
                              f"({most_affected_component[1]} 个差异)")
        conclusion_parts.append(f"🔍 **影响最大的层级**: Layer {most_affected_layer[0]} "
                              f"({most_affected_layer[1]} 个差异)")
        
        # 分析最严重的差异
        if patterns['worst_differences']:
            worst = patterns['worst_differences'][0]
            if worst['relative_diff'] > 0.1:  # 10%以上的差异
                conclusion_parts.append(f"🚨 **最严重差异**: {worst['record_key']} 的 {worst['metric']} "
                                      f"指标相对差异达到 {worst['relative_diff']:.2%}")
    
    # 步骤分析
    step_with_most_diffs = max(step_results, key=lambda x: x['differences'])
    if step_with_most_diffs['differences'] > 0:
        conclusion_parts.append(f"📊 **Step {step_with_most_diffs['step']}** 差异最多 "
                              f"({step_with_most_diffs['differences']} 个)")
    
    # 建议
    if total_differences == 0:
        conclusion_parts.append("💡 **建议**: 两个版本实现高度一致，可以放心使用")
    elif match_rate >= 95:
        conclusion_parts.append("💡 **建议**: 差异很小，可能是数值精度导致，建议验证计算逻辑")
    else:
        conclusion_parts.append("💡 **建议**: 差异较大，建议检查模型权重加载、计算精度设置和算法实现")
    
    return "\n".join(conclusion_parts)


def main():
    parser = argparse.ArgumentParser(description='比较PyTorch和JAX的debug trace精度')
    parser.add_argument('pytorch_file', help='PyTorch debug trace JSON文件')
    parser.add_argument('jax_file', help='JAX debug trace JSON文件')
    parser.add_argument('--tolerance', type=float, default=1e-5,
                       help='数值比较容差 (默认: 1e-5)')
    parser.add_argument('--max-details', type=int, default=10,
                       help='显示的最大详细差异数量 (默认: 10)')
    parser.add_argument('--show-details', action='store_true',
                       help='显示详细差异信息')
    
    args = parser.parse_args()
    
    print("🔍 PyTorch vs JAX Debug Trace 精度比较")
    print("=" * 80)
    
    # 加载文件
    pytorch_data = load_trace_file(args.pytorch_file)
    jax_data = load_trace_file(args.jax_file)
    
    if not pytorch_data or not jax_data:
        print("❌ 文件加载失败")
        return 1
    
    # 按步骤组织数据
    pt_by_step = extract_records_by_step(pytorch_data)
    jax_by_step = extract_records_by_step(jax_data)
    
    print(f"\n📊 数据概览:")
    print(f"   PyTorch steps: {sorted(pt_by_step.keys())}")
    print(f"   JAX steps: {sorted(jax_by_step.keys())}")
    
    # 获取共同的步骤
    common_steps = set(pt_by_step.keys()) & set(jax_by_step.keys())
    print(f"   共同steps: {sorted(common_steps)}")
    
    if not common_steps:
        print("❌ 没有找到可比较的步骤")
        return 1
    
    # 逐步比较
    step_results = []
    for step in sorted(common_steps):
        result = compare_step_records(
            pt_by_step[step], jax_by_step[step], step, args.tolerance)
        step_results.append(result)
    
    # 分析差异模式
    patterns = analyze_difference_patterns(step_results)
    
    # 输出结果
    print_step_summary(step_results)
    print_match_details(step_results)
    print_pattern_analysis(patterns)
    
    if args.show_details:
        print_detailed_differences(step_results, args.max_details)
    
    # 生成最终结论
    print("\n" + "="*80)
    print("🎯 最终结论")
    print("="*80)
    conclusion = generate_conclusion(step_results, patterns, args.tolerance)
    print(conclusion)
    
    # 统计总结
    total_comparisons = sum(r['total_comparisons'] for r in step_results)
    total_differences = sum(r['differences'] for r in step_results)
    
    print(f"\n📈 统计总结:")
    print(f"   总比较次数: {total_comparisons}")
    print(f"   总差异数量: {total_differences}")
    print(f"   使用容差: {args.tolerance}")
    
    return 0 if total_differences == 0 else 1


if __name__ == "__main__":
    sys.exit(main()) 