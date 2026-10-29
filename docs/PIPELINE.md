# Pipeline and design decisions

This document explains the method behind the code: what the model sees, how it
is trained, how the checkpoint is selected, and which engineering details make
the result reproducible.

## 1. Problem formulation

Given an RGB image `x`, predict one of six landmark classes:

```text
Brooklyn_Bridge, Eiffel_Tower, Golden_Gate,
Great_Wall, Sagrada_Familia, Tokio_Tower
```

This is single-label, closed-world image classification. The model outputs six
logits; softmax converts them to probabilities at inference time.

## 2. Data

| Split | Images | Purpose |
|---|---:|---|
| Training pool (`data/train`) | 3,040 | Fitting the model |
| Validation (25% of the pool) | 760 | Checkpoint selection |
| Training (75% of the pool) | 2,280 | Gradient updates |
| Test (`data/test`) | 95 | Final, single evaluation |

The split inside `data/train` is stratified by class and seeded, so every class
appears in both train and validation. The test set is never used for model
selection.

The training images deliberately vary in season, time of day, viewpoint, zoom,
colour treatment and style. The test set is harder: Instagram screenshots with
people, text overlays, filters and crops. This domain shift is the main reason
validation accuracy (~96.7%) is higher than test accuracy (94.7%).

## 3. Preprocessing and augmentation

### Evaluation

1. Convert to RGB (handles grayscale and RGBA inputs).
2. Resize the shorter side to `round(224 × 256 / 224) = 256`.
3. Center-crop to 224 × 224.
4. Scale pixels to `[0, 1]` and normalize:

```text
x' = (x - mean) / std
```

`mean` and `std` are read from `timm.data.resolve_model_data_config`, so they
match the pretrained backbone exactly. For the ImageNet models used here they
are the usual ImageNet statistics, but the code does not assume that.

### Training

```text
RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.75, 1.33))
RandomHorizontalFlip(p=0.5)
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)
ToTensor + Normalize
```

The crop range teaches scale and translation invariance. The flip is safe for
architecture (a mirrored Eiffel Tower is still the Eiffel Tower). The mild
colour jitter covers different lighting and filters without changing the
landmark's identity.

## 4. Model

```text
image (3 × 224 × 224)
  └─ MobileNetV3-Large backbone (ImageNet-1k pretrained, num_classes=0)
       └─ 1,280-dimensional pooled feature vector
            └─ Dropout (optional)
                 └─ Linear(1,280 → 6)
                      └─ logits
```

`num_classes=0` tells timm to remove the original 1,000-way classifier and
return the pooled features. The new linear layer is the only randomly
initialized part of the model.

The backbone is created with `pretrained=True`; if a different timm model is
selected, its feature dimension and preprocessing contract are discovered
automatically.

## 5. Two-phase training

### Phase 1 — head only

The backbone is frozen. Only the new linear classifier is optimized.

| Setting | Value |
|---|---|
| Epochs | 8 |
| Optimizer | Adam |
| Learning rate | `1e-3` |
| Weight decay | 0.0 |
| Batch size | 64 |
| Loss | Cross-entropy |

This is a linear probe: it finds a useful decision boundary in the pretrained
feature space without modifying the features.

### Phase 2 — partial fine-tuning

The best phase-1 weights are restored, then the last two backbone stages and
the classifier are unfrozen.

| Setting | Value |
|---|---|
| Epochs | 5 |
| Optimizer | Adam |
| Learning rate | `1e-4` |
| Unfrozen stages | `blocks[-2:]` |
| Trainable | Last stages + classifier |

The smaller learning rate prevents the pretrained filters from being destroyed
by large updates. Earlier stages stay frozen because they encode generic
low-level features (edges, textures) that are still useful.

### Why not full fine-tuning?

Full fine-tuning on 2,280 training images would give the network enough
capacity to memorize the training set, especially with a 5.5M-parameter
backbone. Partial fine-tuning keeps most of the pretrained representation
intact while adapting the high-level, task-specific stages.

## 6. Loss and optimization

The model returns logits, not probabilities. `CrossEntropyLoss` combines
`log_softmax` and negative log-likelihood internally:

```text
L = -log( softmax(z)[y] )
```

No softmax is added inside the model. At inference, softmax is applied
explicitly to obtain probabilities. This avoids the common mistake of applying
softmax twice (once in the model and once in the loss).

## 7. Validation and checkpointing

After every epoch:

1. Compute validation loss and accuracy.
2. If validation loss improved, copy the model state to CPU memory.
3. At the end of the phase, restore the best state.
4. Phase 2 starts from the best phase-1 state.
5. After both phases, the better of the two best states is used for test.

The original course scripts had a subtle bug: the "best" comparison happened
after the epoch loop, so the last epoch was always saved, and phase 2
overwrote phase 1 unconditionally. This version fixes both issues. It also
saves the class names and preprocessing statistics in the checkpoint, which
makes standalone inference possible.

### Frozen BatchNorm

`model.train()` switches every BatchNorm layer to training mode. If the
backbone is frozen, its running mean/variance would still drift on the new
dataset. After each `model.train()`, the training loop calls
`set_frozen_batchnorm_eval`, which puts BatchNorm layers with no trainable
affine parameters back into eval mode. In phase 2, BatchNorm inside the
unfrozen blocks stays in training mode.

## 8. Evaluation

The test split is evaluated once, after model selection:

- **Loss** — sample-weighted mean cross-entropy.
- **Accuracy** — correct / total.
- **Per-class recall** — correct for class `c` / all samples of class `c`.
- **Macro F1** — unweighted mean of per-class F1, so rare classes count as much
  as frequent ones.
- **Confusion matrix** — rows are true labels, columns are predictions.

Metrics are accumulated over the whole split, not averaged over batches, so the
last partial batch does not get the same weight as a full batch.

## 9. Reproducibility and artifacts

The training run writes:

```text
runs/<run>/
├── best.pt
├── config.json
├── history.json
├── training_curves.png
├── test_metrics.json
├── test_confusion_matrix.png
├── test_classification_report.txt
└── test_predictions.csv
```

`best.pt` contains:

```text
model_name, class_names, image_size, mean, std,
dropout, phase, epoch, val_loss, test_metrics,
config, format_version, state_dict
```

Prediction, evaluation and ONNX export all reconstruct the model from this
metadata instead of relying on hard-coded values.

## 10. Improvements over the original scripts

The original P3 folder contained three near-duplicate training scripts and a
separate notebook. While turning it into a portfolio repository, the following
changes were made:

| Original | Portfolio version |
|---|---|
| Hard-coded Windows save path | `--output-dir`, portable `Path` |
| Random 75/25 split | Stratified, seeded split |
| Best-loss comparison after the epoch loop | Best state tracked and restored per epoch |
| Phase 2 overwrote phase 1 | Best phase-1 weights restored before phase 2 |
| Final model = last phase-2 epoch | Best checkpoint across both phases |
| Dummy forward could update BatchNorm stats | Uses `backbone.num_features`; fallback in eval mode |
| Per-batch accuracy averaged | Sample-weighted metrics |
| Resize 256 hard-coded | Derived from the backbone input size |
| Three duplicate scripts | One CLI with `--model` |
| No metadata in checkpoint | Class names + preprocessing + config stored |
| No deployment path | ONNX export + ONNX Runtime inference |

These are engineering improvements, not changes to the core method. The
two-phase transfer-learning idea, the architecture and the augmentation policy
remain those of the original project.

## 11. Limitations

- The test set has 95 images; one error is about 1.05 percentage points.
- The images are web-scraped and not a curated benchmark.
- The model is closed-world: it has no "unknown" class.
- Softmax confidence is not calibrated.
- The repository contains no robot control or simulator code; it is the
  perception module only.
