# Recorded results

These JSON files contain the confusion matrices from the final 95-image
held-out test set for the two backbones that were compared:

| File | Model |
|---|---|
| `mobilenetv3_metrics.json` | `mobilenetv3_large_100.ra_in1k` (selected model) |
| `efficientnet_lite0_metrics.json` | `tf_efficientnet_lite0.in1k` |

The matrices are the source of the numbers in the README and in
[`docs/RESULTS.md`](../docs/RESULTS.md). Rows are true labels and columns are
predicted labels. The class order is the alphabetical order produced by
`ImageFolder`.

`scripts/plot_results.py` turns these matrices into the published figures:

```bash
python scripts/plot_results.py
```

The training script and `evaluate.py` write the same class order and confusion
matrix automatically (`test_metrics.json`), so a new run can replace these
files without changing the plotting code.

**Why matrices instead of a metrics dump?** The original run was executed in a
course environment and the raw prediction log was not preserved. The confusion
matrices were, so they are the most reliable record of the final evaluation.
Small differences on retraining are expected and are discussed in
[`docs/RESULTS.md`](../docs/RESULTS.md).
