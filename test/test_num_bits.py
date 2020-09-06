import numpy as np
import torch
import tensorflow as tf


def Num2Bit(Num, B):
    Num_ = tf.cast(Num, tf.int32)

    def integer2bit(integer, num_bits=B * 2):
        exponent_bits = tf.range((num_bits - 1), -1, -1, dtype=tf.int32)
        out = tf.expand_dims(integer, -1) // 2 ** exponent_bits
        return (out - (out % 1)) % 2

    bit = integer2bit(Num_)
    bit = tf.reshape(bit[:, :, B:], [-1, Num_.shape[1] * B])
    return tf.cast(bit, tf.float32)


def Bit2Num(Bit, B):
    Bit_ = tf.cast(Bit, tf.float32)
    Bit_ = tf.reshape(Bit_, [-1, Bit_.shape[1] // B, B])
    exponent_bits = tf.range((B * 2 - 1), -1, -1, dtype=tf.int32)[B:]
    num = tf.reduce_sum(Bit_ * tf.cast(2 ** exponent_bits, tf.float32), -1)
    return num


@tf.custom_gradient
def QuantizationOp(x, B):
    step = tf.cast((2 ** B), dtype=tf.float32)

    result = tf.cast((tf.round(x * step - 0.5)), dtype=tf.float32)

    result = Num2Bit(result, B)

    def custom_grad(dy):
        grad = tf.reduce_sum(tf.reshape(dy, [tf.shape(dy)[0], -1, B]), axis=2)
        return grad, None

    return result, custom_grad


@tf.custom_gradient
def DequantizationOp(x, B):
    step = tf.cast((2 ** B), dtype=tf.float32)
    out = Bit2Num(x, B)
    result = (out + 0.5) / step

    def custom_grad(dy):
        grad_bit = tf.tile(tf.expand_dims(dy, -1), [1, 1, B])
        # grad_bit = tf.transpose(grad_bit, [0, 2, 1])
        grad = tf.reshape(grad_bit, [tf.shape(grad_bit)[0], -1])
        return grad, None

    return result, custom_grad


def Num2Bit_t(Num, B):
    Num_ = Num.type(torch.uint8)

    def integer2bit(integer, num_bits=B * 2):
        dtype = integer.type()
        exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
        exponent_bits = exponent_bits.repeat(integer.shape + (1,))
        out = integer.unsqueeze(-1) // 2 ** exponent_bits
        return (out - (out % 1)) % 2

    bit = integer2bit(Num_)
    bit = (bit[:, :, B:]).reshape(-1, Num_.shape[1] * B)
    return bit.type(torch.float32)


def Bit2Num_t(Bit, B):
    Bit_ = Bit.type(torch.float32)
    Bit_ = torch.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])
    num = torch.zeros(Bit_[:, :, 1].shape)
    for i in range(B):
        num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)
    return num


class Quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = torch.round(x * step - 0.5)
        out = Num2Bit_t(out, B)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of constant arguments to forward must be None.
        # Gradient of a number is the sum of its four bits.
        b, _ = grad_output.shape
        grad_num = torch.sum(grad_output.reshape(b, -1, ctx.constant), dim=2)
        return grad_num, None


class Dequantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = Bit2Num_t(x, B)
        out = (out + 0.5) / step
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        # repeat the gradient of a Num for four time.
        # b, c = grad_output.shape
        # grad_bit = grad_output.repeat(1, 1, ctx.constant)
        # return torch.reshape(grad_bit, (-1, c * ctx.constant)), None
        grad_bit = grad_output.repeat_interleave(ctx.constant, dim=1)
        return grad_bit, None

B = 4

x = tf.random.uniform((2, 32), 0, 1)
w = tf.random.normal((2, 32))
with tf.GradientTape() as tape:
    tape.watch(x)
    out1 = QuantizationOp(x, B)
    out2 = DequantizationOp(out1, B)
    loss = tf.reduce_mean(out2 * w)
grads = tape.gradient(loss, [x])[0]


xt = torch.from_numpy(x.numpy()).requires_grad_()
xt.grad = None
wt = torch.from_numpy(w.numpy())
out_t1 = Quantization.apply(xt, B)
out_t2 = Dequantization.apply(out_t1, B)
loss_t = (out_t2 * wt).mean()
loss_t.backward()

np.testing.assert_equal(out1.numpy(), out_t1.detach().numpy())
np.testing.assert_equal(out2.numpy(), out_t2.detach().numpy())
np.testing.assert_equal(grads.numpy(), xt.grad.numpy())
