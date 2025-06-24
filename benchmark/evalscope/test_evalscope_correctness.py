import argparse
import yaml
import sys
from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType
import datetime
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load YAML configuration file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        sys.exit(1)

def run_evaluation(test_suite):
    """Run evaluation for a single test suite"""
    model_name = test_suite.get('model_name')
    base_url = test_suite.get('base_url')
    tasks = test_suite.get('tasks', [])
    
    logger.info(f"Starting evaluation for model: {model_name}, API URL: {base_url}")
    
    all_results = {}
    all_passed = True
    
    for task in tasks:
        task_name = task.get('name')
        dataset = task.get('dataset')
        metrics_targets = task.get('metrics', [])
        use_fewshot = task.get('use_fewshot', False)
        num_fewshot = task.get('num_fewshot', 0)
        use_sample = task.get('use_sample', False)
        sample_num = task.get('sample_num', 100)
        max_concurrency = task.get('max_concurrency', 32)
        
        logger.info(f"Evaluating task: {task_name}, dataset: {dataset}")
        
        task_config = {
            'model': model_name,
            'api_url': base_url,
            'api_key': 'EMPTY',
            'eval_type': EvalType.SERVICE,
            'datasets': [dataset],
            'eval_batch_size': max_concurrency,
            'generation_config': {
                'temperature': 0.0,
            }
        }

        if use_sample:
            task_config['limit'] = sample_num
        
        if use_fewshot:
            dataset_args = {
                dataset: {
                    "few_shot_num": num_fewshot,
                    "few_shot_random": False,
                }
            }
            task_config['dataset_args'] = dataset_args
            logger.info(f"Using few-shot configuration: {num_fewshot} samples")
        
        task_cfg = TaskConfig(**task_config)
        
        try:
            results = run_task(task_cfg=task_cfg)
            all_results[task_name] = results
            
            if dataset in results:
                metrics = results[dataset]
                
                logger.info(f"Task {task_name} results: {metrics}")
                
                for metric_target in metrics_targets:
                    target_value = metric_target.get('target')
                    tolerance = metric_target.get('tolerance', 0.0)
                    
                    if isinstance(metrics, dict):
                        if 'accuracy' in metrics:
                            actual_value = metrics['accuracy']
                        else:
                            actual_value = next(iter(metrics.values()))
                    else:
                        if hasattr(metrics, 'score'):
                            actual_value = metrics.score
                        else:
                            actual_value = metrics
                    
                    lower_bound = target_value - tolerance
                    upper_bound = target_value + tolerance
                    passed = lower_bound <= actual_value <= upper_bound
                    
                    logger.info(f"Metric check: target={target_value}, actual={actual_value}, tolerance={tolerance}, passed={passed}")
                    
                    if not passed:
                        all_passed = False
            else:
                logger.error(f"Results not found for dataset {dataset}")
                all_passed = False
        except Exception as e:
            logger.error(f"Failed to evaluate task {task_name}: {e}")
            all_passed = False
    
    return all_passed, all_results

def main():
    parser = argparse.ArgumentParser(description="Model Accuracy Testing Tool")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    test_suites = config.get('test_suites', [])
    
    if not test_suites:
        logger.error("No test suites found in config file")
        sys.exit(1)
    
    all_suites_passed = True
    for test_suite in test_suites:
        suite_passed, results = run_evaluation(test_suite)
        if not suite_passed:
            all_suites_passed = False
    
    if all_suites_passed:
        logger.info("All test suites passed!")
        sys.exit(0)
    else:
        logger.error("Some test suites failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
