# Model file
Model files may contain model weights, optimizer states and other training states, for resuming training or transfer learning.

## Format
`hanser` has three formats of model files. Although with different names, the file content is all TensorFlow checkpoint.

### A. Training checkpoint
Training checkpoint is produced by model training (e.g. Learner.save).
This format contains model weights, optimizer states and epoch number.
We usually use this format to save and resume training.

### B. Model checkpoint
Model checkpoint is an intermediate format and contains only model weights.
This format has smaller size. The users may not use this format directly.

### C. Model file
Model file is an archive (usually .zip) of model checkpoint, with PyTorch style name.
This format is easier for distribution than original TensorFlow checkpoint, which consists of many files.
It is the most widely used format in `hanser`. We use this format to do transfer learning.


## Conversion

### A -> C
```bash
python snippets/release_model.py ~/Downloads/83/ckpt resnetvd50 ~/Downloads
```

### A -> B (internal)
```python
from hanser.models.utils import convert_checkpoint
convert_checkpoint("~/Downloads/83/ckpt", "~/Downloads/83/model")
```