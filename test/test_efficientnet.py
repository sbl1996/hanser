from hanser.models.imagenet.efficientnet import efficientnet_b2
from hanser.models.profile.fvcore import profile
model = efficientnet_b2()
model.build((None, 456, 456, 3))
profile(model)
