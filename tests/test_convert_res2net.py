import numpy as np
import tensorflow as tf

from hanser.models.legacy.imagenet.res2net.resnet_vd import resnet50 as resnet50_1
from hanser.models.imagenet.res2net.resnet_vd import resnet50 as resnet50_2
from hanser.models.utils import load_pretrained_model, convert_checkpoint

def convert_res2conv(gconv, convs):
    gbn = gconv.layers[1]
    gconv = gconv.layers[0].layers[1]
    bns = [conv.layers[1] for conv in convs]
    convs = [conv.layers[0].layers[1] for conv in convs]
    g = len(convs)
    c = gconv.kernel.shape[-1] // g
    for i in range(g):
        w = gconv.kernel[..., i*c:(i+1)*c]
        convs[i].kernel.assign(w)
        bn = bns[i]
        bn.gamma.assign(gbn.gamma[i*c:(i+1)*c])
        bn.beta.assign(gbn.beta[i*c:(i+1)*c])
        bn.moving_mean.assign(gbn.moving_mean[i*c:(i+1)*c])
        bn.moving_variance.assign(gbn.moving_variance[i*c:(i+1)*c])


def convert_bottleneck(block1, block2):
    block2.conv1.set_weights(block1.conv1.get_weights())
    convert_res2conv(block1.conv2.conv, block2.conv2.convs)
    block2.conv3.set_weights(block1.conv3.get_weights())
    block2.bn3.set_weights(block1.bn3.get_weights())
    block2.shortcut.set_weights(block1.shortcut.get_weights())


def convert_res2net(net1, net2):
    net2.stem.set_weights(net1.stem.get_weights())
    for i in range(4):
        layer_name = "layer" + str(i + 1)
        net1_layer = getattr(net1, layer_name)
        net2_layer = getattr(net2, layer_name)
        convert_bottleneck(net1_layer.layers[0], net2_layer.layers[0])
        for i in range(1, len(net1_layer.layers)):
            net2_layer.layers[i].set_weights(net1_layer.layers[i].get_weights())
    net2.fc.set_weights(net1.fc.get_weights())


x = np.load("/Users/hrvvi/Downloads/cat1_input.npy")
y = np.load("/Users/hrvvi/Downloads/cat1_logits.npy")

net1 = resnet50_1()
net1.build((None, 224, 224, 3))
load_pretrained_model('legacy_res2netvd50_nlb', net1, with_fc=True)

net2 = resnet50_2()
net2.build((None, 224, 224, 3))

convert_res2net(net1, net2)

xt = tf.convert_to_tensor(x)
yt = net2(x[None])[0].numpy()
np.testing.assert_allclose(yt, y, atol=1e-5)

ckpt = tf.train.Checkpoint(model=net2)
ckpt_path = "/Users/hrvvi/Downloads/res2net/ckpt"
ckpt.write(ckpt_path)
convert_checkpoint(ckpt_path, "/Users/hrvvi/Downloads/res2net/model")
# python snippets/release_model.py "/Users/hrvvi/Downloads/res2net/model" res2netvd50_nlb "/Users/hrvvi/Downloads"