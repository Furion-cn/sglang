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