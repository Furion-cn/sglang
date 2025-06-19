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
    "cmmlu": {
        "few_shot_num": 5,
        "few_shot_random": False,
    },
    "mmlu": {
        "few_shot_num": 5,
        "few_shot_random": False,
    },
    "ceval": {
        "few_shot_num": 5,
        "few_shot_random": False,
    },
    "humaneval": {
        "few_shot_num": 0,
        "few_shot_random": False,
    },
    "competition_math": {
        "few_shot_num": 4,
        "few_shot_random": False,
    },
    "bbh": {
        "few_shot_num": 3,
        "few_shot_random": False,
    }
}

def benchmark(args, datasets):
    # 初始化swanlab run来跟踪这次评测
    swanlab.login(api_key=args.swanlab_api_key)
    experiment_name = f"{args.model}-{args.inference_engine_name}-{args.inference_engine_version}-{args.limit}"
    tags = [
        f"model:{args.model}",
        f"inference_engine:{args.inference_engine_name}",
        f"inference_engine_version:{args.inference_engine_version}",
        f"num_samples:{args.limit}",
        f"max_concurrency:{args.max_concurrency}",
        f"few_shot:{args.few_shot}",
    ]
    for dataset in datasets:
        tags.append(f"dataset:{dataset}")
    
    experiment_name = f"{args.model}-{args.inference_engine_name}-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    swanlab.init(
        project="qwen-7b",
        workspace="Furion",
        experiment_name=experiment_name,
        config={
            "model": args.model,
            "api_url": args.api_url,
            "inference_engine": args.inference_engine_name,
            "inference_engine_version": args.inference_engine_version,
            "num_samples": args.limit,
            "max_concurrency": args.max_concurrency,
            "datasets": datasets,
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
        'datasets': datasets,
        'limit': args.limit,
        'eval_batch_size': args.max_concurrency,
        'generation_config': {
            'temperature': 0.0,
        }
    }
    
    # use few-shot config
    if args.few_shot:
        dataset_args = {}
        for dataset in datasets:
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
        
        for dataset_name, metrics in results.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    log_key = f"{dataset_name}_{metric_name}"
                    log_data[log_key] = value
                    
                if 'accuracy' in metrics:
                    log_data[f"{dataset_name}_score"] = metrics['accuracy'] * 100
            else:
                if hasattr(metrics, 'score'):
                    log_data[f"{dataset_name}_score"] = metrics.score * 100
                else:
                    log_data[f"{dataset_name}_score"] = metrics
        
        swanlab.log(log_data, print_to_console=True, step=args.limit)
        
        print("评测结果已记录到SwanLab")    
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
    parser.add_argument(
        "--datasets",
        nargs='+',
        default=['gsm8k', 'cmmlu', 'mmlu', 'ceval', 'humaneval', 'competition_math', 'bbh'],
        help="Datasets to evaluate, can specify multiple times (e.g., --datasets gsm8k arc)",
    )

    args = parser.parse_args()
    benchmark(args, datasets=args.datasets)