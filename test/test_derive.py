from hhutil.io import read_pickle

import tensorflow as tf
from hanser.models.nas.darts.model_search_pc_darts import beta_softmax
from hanser.models.nas.genotypes import get_primitives, Genotype


def derive(alphas_normal, alphas_reduce, betas_normal, betas_reduce,
           primitives, steps=4, multiplier=4, max_sk=2):
    def get_op(w):
        if 'none' in primitives:
            i = max([k for k in range(len(primitives)) if k != primitives.index('none')], key=lambda k: w[k])
        else:
            i = max(range(len(primitives)), key=lambda k: w[k])
        return w[i], primitives[i]

    def _parse(alphas):
        gene = []
        start = 0
        for i in range(steps):
            end = start + i + 2
            W = alphas[start:end]
            edges = sorted(range(i + 2), key=lambda x: -get_op(W[x])[0])[:2]
            for j in edges:
                gene.append((get_op(W[j])[1], j))
            start = end
        return gene

    def _parse_max_sk(alphas):
        gene = _parse(alphas)
        if max_sk:
            sk_idx = primitives.index('skip_connect')
            while len([op for op in gene if op[0] == 'skip_connect']) > max_sk:
                print(alphas)
                print(len([op for op in gene if op[0] == 'skip_connect']))
                sk_probs = alphas[:, sk_idx].copy()
                sk_probs[sk_probs == 0] = 1
                alphas[sk_probs.argmin(), sk_idx] = 0.
                gene = _parse(alphas)
        return gene

    alphas_normal = tf.nn.softmax(alphas_normal, axis=-1).numpy()
    betas_normal = beta_softmax(betas_normal, steps, scale=True).numpy()
    alphas_normal = alphas_normal * betas_normal[:, None]

    alphas_reduce = tf.nn.softmax(alphas_reduce, axis=-1).numpy()
    betas_reduce = beta_softmax(betas_reduce, steps, scale=True).numpy()
    alphas_reduce = alphas_reduce * betas_reduce[:, None]

    gene_normal = _parse_max_sk(alphas_normal)
    gene_reduce = _parse_max_sk(alphas_reduce)

    concat = range(2 + steps - multiplier, steps + 2)
    genotype = Genotype(
        normal=gene_normal, normal_concat=concat,
        reduce=gene_reduce, reduce_concat=concat
    )
    return genotype


alphas_normal, alphas_reduce, betas_normal, betas_reduce = [
    tf.convert_to_tensor(p) for p in read_pickle("/Users/hrvvi/Downloads/arch (1).pickle")
]

steps = 4
# alphas_normal = tf.nn.softmax(alphas_normal, axis=-1).numpy()
# betas_normal = beta_softmax(betas_normal, steps).numpy()
# alphas_reduce = tf.nn.softmax(alphas_reduce, axis=-1).numpy()
# betas_reduce = beta_softmax(betas_reduce, steps).numpy()
primitives = get_primitives()

derive(alphas_normal, alphas_reduce, betas_normal, betas_reduce, primitives)
