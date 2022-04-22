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

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            stage = item
            return self._history[stage]
        if len(item) == 2:
            stage, metric = item
            return self.get_metric(metric, stage)
        elif len(item) == 3:
            stage, metric, range = item
            if isinstance(range, int):
                start, end = range, range
            elif isinstance(range, slice):
                start, end = range.start, range.stop - 1
            else:
                raise ValueError(f"Invalid range: {range}")
            return self.get_metric(metric, stage, start, end)

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
            if start == -1:
                start = max_epochs
            if end is None or end == -1:
                end = max_epochs
            values = []
            for e in range(start, end + 1):
                if e in h:
                    values.append(h[e].get(metric))
            if all(v is None for v in values):
                return None
            elif start == end:
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