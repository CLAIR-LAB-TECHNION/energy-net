import os
def get_model(model=None,is_recurrent=False,is_td3=False):
    # Default best model and norm paths
    if model:
        model = model
    else:
        # Check for algorithm-specific best models first
        if is_recurrent:
            prefix = "rppo"
        elif is_td3:
            prefix = "td3"
        else:
            prefix = "ppo"
        model = f"{prefix}_iso_best.zip"

        # Fall back to legacy model path if not found
        # Fall back to legacy model path if not found
        if not os.path.exists(model):
            model = "iso_best.zip"

    return model