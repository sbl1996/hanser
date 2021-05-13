from hanser.models.transformer.vision.vit import VisionTransformer

def deit_ti(**kwargs):
    return VisionTransformer(
        d_model=192, num_heads=3, num_layers=12, dff=768, **kwargs)

def deit_s(**kwargs):
    return VisionTransformer(
        d_model=384, num_heads=6, num_layers=12, dff=1536, **kwargs)

def deit_b(**kwargs):
    return VisionTransformer(
        d_model=768, num_heads=12, num_layers=12, dff=3072, **kwargs)