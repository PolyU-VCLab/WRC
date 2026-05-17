from omegaconf import OmegaConf


def get_feature(target: str) -> int:
    """Resolve feature dimensions from backbone name patterns"""
    model_name = target.lower()

    if "vits" in model_name or "vit_small" in model_name:
        return 384
    if "vitb" in model_name or "vit_base" in model_name:
        if model_name == 'vit_base_patch16_clip':
            return 512
        else:
            return 768
    if "vitl" in model_name or "vit_large" in model_name:
        return 1024
    if model_name == "efficientnet_b4":
        return 128
    if model_name == "maskclip":
        return 512
    if model_name == "radio_v2.5-h":
        return 1280
    if model_name == "radio_v2.5-l":
        return 1024
    if model_name == "radio_v2.5-b":
        return 768

    raise ValueError(f"Unsupported backbone: {model_name}")


OmegaConf.register_new_resolver("get_feature", lambda name: get_feature(name))


def get_patch(target: str) -> int:
    """Resolve patch from backbone name patterns"""
    model_name = target.lower()

    if "patch14" in model_name:
        return 14
    if "patch16" in model_name:
        return 16

    raise ValueError(f"Unsupported backbone: {model_name}")


OmegaConf.register_new_resolver("get_patch", lambda name: get_patch(name))
