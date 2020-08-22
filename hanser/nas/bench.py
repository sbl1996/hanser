import torch


def simplify(d):
    min_keys = {
        'eval_acc1es': 'ori-test@199',
        'eval_losses': 'ori-test@199',
        'eval_times': 'ori-test@199',
        'train_acc1es': 199,
        'train_acc5es': None,
        'train_losses': 199,
        'train_times': 199
    }
    for i in range(d['total_archs']):
        print(i)
        x = d['arch2infos'][i]
        del x['less']
        for dk, res in x['full']['all_results'].items():
            for k, v in res.items():
                if k in min_keys:
                    mk = min_keys[k]
                    if mk:
                        res[k] = v[mk]


class SimpleNASBench201:

    def __init__(self, fp):
        self.d = torch.load(fp)

        self.arch2index = {}
        for i, arch in enumerate(self.d['meta_archs']):
            self.arch2index[arch] = i

    def query_all(self, arch_or_index, dataset='cifar10'):
        if isinstance(arch_or_index, str):
            index = self.arch2index[arch_or_index]
        else:
            index = arch_or_index
        x = self.d['arch2infos'][index]['full']
        seeds = x['dataset_seed'][dataset]
        results = {}
        for seed in seeds:
            results[seed] = x['all_results'][(dataset, seed)]
        return results

    def query_eval_acc(self, arch_or_index, dataset='cifar10'):
        results = self.query_all(arch_or_index, dataset)
        return [r['eval_acc1es'] for r in results.values()]
