import tensorflow as tf

def scaled_dot_product_attention(q, k, v):
    # q: (N, H*W, C)
    # (N, H, W, C)
    # (N, H/M, M, W/M, M, C)
    # (N, H/M, W/M, M, M, C)
    # (N, H/M, W/M, M*M, C)
    # attention
    # (N, H/M*W/M, M*M, C)

    # q: (N, H*W, C)
    # k: (N, H*W, C)
    # v: (N, H*W, C)
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], matmul_qk.dtype)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, v)
    return output, attention_weights

