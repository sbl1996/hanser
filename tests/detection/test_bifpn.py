import tensorflow as tf
from hanser.models.detection.bifpn import BiFPNCell, BiFPN

f = 160
xs = [tf.random.normal((2, s, s, c)) for s, c in zip([80, 40, 20], [512, 1024, 2048])]

cs = [x.shape[-1] for x in xs]
# fpn = BiFPNCell(cs, f, seperable_conv=True)
fpn = BiFPN(cs, f, repeats=6, seperable_conv=False)
ps = fpn(xs)
sum([w.shape.num_elements() for w in fpn.weights])