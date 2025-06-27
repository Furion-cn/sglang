import json
from sglang.srt.jax.model_converter import converter_logging

def str2bool(v: str) -> bool:
  v = v.lower()
  true_values = ["y", "yes", "t", "true", "1"]
  false_values = ["n", "no", "f", "false", "0"]
  if v in true_values:
    return True
  elif v in false_values:
    return False
  else:
    raise ValueError(f"Invalid value '{v}'!")

def sed_model_config(config_file: str):
    """
    Modify the architectures field in the config.json file.
    Rules:
    - If the architecture ends with "Model", replace "Model" with "JaxModel"
    - Otherwise, add "JaxModel" to the end of the architecture
    """
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        if 'architectures' not in config:
            converter_logging.log(f"⚠️  No 'architectures' field found in {config_file}")
            return
            
        original_architectures = config['architectures']
        new_architectures = []
        
        converter_logging.log(f"Original architectures: {original_architectures}")
        
        for arch in original_architectures:
            if arch.endswith('Model'):
                new_arch = arch[:-5] + 'JaxModel' 
                new_architectures.append(new_arch)
                converter_logging.log(f"Modified: {arch} → {new_arch}")
            else:
                new_arch = arch + 'JaxModel'
                new_architectures.append(new_arch)
                converter_logging.log(f"Extended: {arch} → {new_arch}")
        
        config['architectures'] = new_architectures
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            
        converter_logging.log(f"✅ Updated architectures in {config_file}")
        converter_logging.log(f"New architectures: {new_architectures}")
        
    except Exception as e:
        converter_logging.log(f"❌ Failed to modify config file {config_file}: {str(e)}")