def remove_prefix(state_dict, prefix):
    """Old style model is stored with all names of parameters sharing common prefix 'module.'"""
    print(f"remove prefix '{prefix}'")

    def helper(x):
        return x.split(prefix, 1)[-1] if x.startswith(prefix) else x

    return {helper(key): value for key, value in state_dict.items()}
