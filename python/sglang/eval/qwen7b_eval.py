from evalscope import TaskConfig, run_task  
from evalscope.constants import EvalType 
import argparse
import swanlab
import datetime

few_shot_config = {
    "gsm8k": {
        "few_shot_num": 4, # GSM8K uses 4-shot examples with CoT or 0-shot by system
        "few_shot_random": False,
    },
    "arc": {
        "few_shot_num": 0, # suggest using 0-shot by system
    },
}

def benchmark(args):
    # 初始化swanlab run来跟踪这次评测
    swanlab.login(api_key=args.swanlab_api_key)
    swanlab.init(
        # 设置项目信息
        project="qwen-7b",
        workspace="Furion",
        # 跟踪评测配置和运行元数据
        config={
            "model": args.model,
            "api_url": args.api_url,
            "inference_engine": args.inference_engine_name,
            "inference_engine_version": args.inference_engine_version,
            "eval_limit": args.limit,
            "max_concurrency": args.max_concurrency,
            "datasets": ['gsm8k', 'arc'],
            "eval_type": "SERVICE",
            "few_shot": args.few_shot,
            "timestamp": datetime.datetime.now().isoformat(),
        }
    )

    # 构建基础配置
    task_config = {
        'model': args.model, 
        'api_url': args.api_url, 
        'api_key': 'EMPTY',  
        'eval_type': EvalType.SERVICE,  
        'datasets': ['gsm8k', 'arc'],
        'limit': args.limit,
        'eval_batch_size': args.max_concurrency,
    }
    
    # 如果启用few-shot，添加dataset_args配置
    if args.few_shot:
        dataset_args = {}
        for dataset in ['gsm8k', 'arc']:
            if dataset in few_shot_config:
                dataset_args[dataset] = few_shot_config[dataset]
        task_config['dataset_args'] = dataset_args
        print(f"启用Few-shot配置: {dataset_args}")
    
    task_cfg = TaskConfig(**task_config)
    
    print("开始评测...")
    results = run_task(task_cfg=task_cfg)
    print("评测完成，结果:", results)
    
    # 记录评测结果到swanlab
    if results:
        log_data = {}
        
        # 解析并记录各个数据集的结果
        for dataset_name, metrics in results.items():
            if isinstance(metrics, dict):
                # 为每个数据集的指标添加前缀
                for metric_name, value in metrics.items():
                    log_key = f"{dataset_name}_{metric_name}"
                    log_data[log_key] = value
                    
                # 如果有accuracy指标，也记录为主要指标
                if 'accuracy' in metrics:
                    log_data[f"{dataset_name}_score"] = metrics['accuracy'] * 100
            else:
                # 如果结果是单个值
                log_data[f"{dataset_name}_score"] = metrics
        
        # 计算平均分数（如果有多个数据集）
        dataset_scores = [v for k, v in log_data.items() if k.endswith('_score')]
        if dataset_scores:
            log_data['average_score'] = sum(dataset_scores) / len(dataset_scores)
        
        # 记录到swanlab
        swanlab.log(log_data)
        
        print("评测结果已记录到SwanLab")
        print(f"记录的指标: {list(log_data.keys())}")
    
    # 完成此次run
    swanlab.finish()
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:30000/v1",
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen-7B",
        help="Model name, ID or path, only used for model name",
    )
    parser.add_argument(
        "--limit", type=int, default=10, help="Number of examples to process",
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=144, help="Maximum concurrent requests",
    )
    parser.add_argument(
        "--few-shot", action="store_true", help="Enable few-shot evaluation",
    )
    parser.add_argument(
        "--inference-engine-name", help="Inference engine name", required=True,
    )
    parser.add_argument(
        "--inference-engine-version", help="Inference engine version", default="default",
    )
    parser.add_argument(
        "--swanlab-api-key", help="SwanLab API key", required=True,
    )
    args = parser.parse_args()
    benchmark(args)