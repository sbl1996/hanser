# Mixup

Mixup would be implemented by 5 different ways, resulting different memory usage, time cost and training diffuculty. CutMix and more mix methods generally follow the same patterns. 

## Concepts

### Training Difficulty

More difficult implementations are usually harder to converge for models. Therefore, overfitted models will have better performance and underfitted models will have worse performance.

### Device

Mixup can be executed in either CPU or TPU.

- CPU
    - Executing on CPU might lead to higher memory usage (out of memory), but is usually fast.
- TPU
    - TODO: Executing on TPU might have lower memory usage and time cost.
    - Beta sampling is not implemented in TPU, we use Monte Carlo sampling to approximate.
    - Mixup on TPU is executed distributedly, which means that different cores have different sampling results, resulting in possible higher training difficulty.

## Implementations

### Batch

Mixup samples within the batch, with the same lambda for every mixed pair, resulting in lowest training difficulty.

The batch will be mixed with its shuffled or reversed batch.

```python
def batch_transform(image, label):
    return mixup_batch(image, label, alpha=0.2)

ds_train, ds_test, steps_per_epoch, test_steps = make_cifar10_dataset(
    batch_size, eval_batch_size, transform, batch_transform=batch_transform)
```

### Batch (TPU)

Just like Batch, but executed on TPU, so different cores have different lambda (same lambda for data on the same core), resulting in higher training difficulty than Batch.

```python
def batch_transform(image, label):
    return mixup_batch(image, label, alpha=0.2, mc=True)

learner = SuperLearner(
    model, criterion, optimizer, batch_transform=batch_transform,
    train_metrics=train_metrics, eval_metrics=eval_metrics,
    work_dir=f"./models")
```

### Batch+

Just like Batch, with different lambda for every mixed pair, resulting higher training difficulty than Batch and Batch (TPU).

```python
def batch_transform(image, label):
    return mixup_batch(image, label, alpha=0.2, hard=True)
```

### Sample

Mixup samples within the whole dataset, but not from the same batch, with the same lambda for every mixed pair. This implementation has significant higher memory usage, time cost, and training diffuculty.

```python
def zip_transform(data1, data2):
    return mixup(data1, data2, alpha=0.2)

ds_train, ds_test, steps_per_epoch, test_steps = make_cifar10_dataset(
    batch_size, eval_batch_size, transform, zip_transform=zip_transform)
```

### Sample+

Just like Sample, with different lambda for every mixed pair, resulting higher training difficulty than Sample.

```python
def zip_transform(data1, data2):
    return mixup(data1, data2, alpha=0.2, hard=True)

ds_train, ds_test, steps_per_epoch, test_steps = make_cifar10_dataset(
    batch_size, eval_batch_size, transform, zip_transform=zip_transform)
```

## Summary
| Implementation | Lambda | Memory | Time | Difficulty |
|----------------|--------|--------|------|------------|
| Batch          | batch  | +      | +    | +          |
| Batch (TPU)    | core   | ?      | ?    | ++         |
| Batch+         | sample | +      | +    | +++        |
| Sample         | batch  | +++    | +++  | +++        |
| Sample+        | sample | +++    | +++  | ++++       |
