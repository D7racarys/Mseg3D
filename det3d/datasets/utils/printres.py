def print_dict_structure(d, indent=0):
    for k, v in d.items():
        vtype = type(v).__name__
        shape_info = ""
        if hasattr(v, "shape"):
            shape_info = f", shape={v.shape}"
        elif isinstance(v, list):
            shape_info = f", len={len(v)}"
        print("  " * indent + f"- {k} ({vtype}{shape_info})")
        if isinstance(v, dict):
            print_dict_structure(v, indent + 1)
