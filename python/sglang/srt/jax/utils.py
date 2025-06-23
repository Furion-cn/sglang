def flatten_pytree_with_paths(pytree, prefix=""):
    flat_dict = {}
    if isinstance(pytree, dict):
        for key, value in pytree.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flat_dict.update(flatten_pytree_with_paths(value, new_prefix))
            else:
                flat_dict[new_prefix] = value
    else:
        flat_dict[prefix] = pytree
    return flat_dict

def get_expected_param_paths(model_state, prefix=""):
    expected_paths = set()
    
    def traverse_state(state, path=""):
        if hasattr(state, 'value'):
            expected_paths.add(path)
        elif isinstance(state, dict):
            for key, value in state.items():
                new_path = f"{path}.{key}" if path else key
                traverse_state(value, new_path)
                
    traverse_state(model_state, prefix)
    return expected_paths

def update_state_recursive(current_state, flat_weights, path=""):
    if hasattr(current_state, 'value'):
        if path in flat_weights:
            current_state.value = flat_weights[path]
        else:
            raise ValueError(f"Missing weight for parameter at path '{path}'")
    elif hasattr(current_state, 'shape'): 
        if path in flat_weights:
            pass
    elif isinstance(current_state, dict):
        for key in current_state:
            new_path = f"{path}.{key}" if path else key
            update_state_recursive(current_state[key], flat_weights, new_path)
    elif hasattr(current_state, '__iter__') and hasattr(current_state, '__getitem__'):
        for key in current_state:
            new_path = f"{path}.{key}" if path else key
            update_state_recursive(current_state[key], flat_weights, new_path)
    else:
        raise TypeError(f"Unsupported state type at path '{path}': {type(current_state)}")