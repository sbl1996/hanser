from tensorflow.keras import Model

class SingleStageDetector(Model):

    def call(self, x):
        xs = self.backbone(x)
        xs = [xs[i] for i in self.backbone_indices]
        xs = self.neck(xs)
        preds = self.head(xs)
        return preds