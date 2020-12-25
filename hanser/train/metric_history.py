class MetricHistory:

    def __init__(self, stages):
        self.stages = stages
        # stage -> epoch -> metric -> value
        self._history = {
            stage: {}
            for stage in stages
        }

    def record(self, stage, epoch, metric, value):
        h = self._history[stage]
        if epoch not in h:
            h[epoch] = {}
        h[epoch][metric] = value

    def get_metric(self, metric, stage=None, start=None, end=None):
        if stage is None:
            return {
                stage: self.get_metric(metric, stage, start, end)
                for stage in self.stages
            }
        else:
            h = self._history[stage]
            epochs = list(h.keys())
            if len(epochs) == 0:
                return None
            min_epoch, max_epochs = min(epochs), max(epochs)
            if start is None:
                start = min_epoch
            if end is None:
                end = max_epochs
            values = []
            for e in range(start, end + 1):
                if e in h:
                    values.append(h[e].get(metric))
            if all(v is None for v in values):
                return None
            elif len(values) == 1:
                return values[0]
            else:
                return values

    def get_epochs(self, start, end, stage=None):
        if stage is None:
            h = {
                stage: self.get_epochs(start, end, stage)
                for stage in self.stages
            }
            for k in h.keys():
                if h[k] is None:
                    del h[k]
            return h
        else:
            h = self._history[stage]
            metrics = set()
            for e in range(start, end + 1):
                if e not in h:
                    continue
                for m in h[e].keys():
                    metrics.add(m)
            return {
                m: self.get_metric(m, stage, start, end)
                for m in metrics
            }