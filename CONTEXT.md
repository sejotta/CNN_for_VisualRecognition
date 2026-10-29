# Portfolio curation notes

This file explains what was kept from the original **P3** folder, what was left
out, and why. It is intentionally separate from the project README: the README
is for visitors, this file is the audit trail of the repository curation.

## What the original project is

The P3 folder is a **six-class landmark image classifier** built with Python,
PyTorch and `timm`:

```text
Brooklyn_Bridge · Eiffel_Tower · Golden_Gate
Great_Wall · Sagrada_Familia · Tokio_Tower
```

The project compares transfer-learning backbones, trains a new classification
head in two phases, evaluates on a held-out Instagram-style test set, and
deploys the model through ONNX for CPU / Raspberry Pi inference.

## Important scope note: no robot / CoppeliaSim code

I searched the P3 folder and the wider course workspace for `coppelia`,
`coppeliasim`, `vrep` and robot-controller files. **No such code exists in
P3.** The project is a standalone computer-vision classifier, not a robot
simulation.

The only deployment-related material in the original project is the
ONNX/Raspberry Pi pipeline (the "low-cost GPU" annex). That process **is**
included in this repository:

- `landmark_classifier/export_onnx.py`
- `deploy/onnx_inference.py`
- `docs/DEPLOYMENT.md`

If there is a separate robot project that should use this classifier, it
belongs in its own repository (or a clearly separated `robotics/` folder) so
the perception code stays reusable. Nothing about CoppeliaSim or the software
used to build the project was uploaded.

## What was included

| Included | Why |
|---|---|
| Clean Python package (`landmark_classifier/`) | The actual portfolio value: readable, modular, reusable code |
| Two-phase training CLI | Reproduces the core method |
| Evaluation, prediction and ONNX export | Shows the full workflow, not just training |
| README, pipeline, results and deployment docs | Explains what the model does and how it works |
| Confusion matrices, training curves, per-class recall | Evidence of results; no personal images |
| Custom licence | Free use for learning + permission for reuse + credit to the teaching team |
| `results/*.json` | Recorded matrices behind the published figures |
| `data/README.md` | Explains the expected layout without redistributing the images |

## What was excluded (and why)

| Excluded | Reason |
|---|---|
| `train_dataset/` (~3,040 images, ~440 MB) | Web-scraped third-party images; large, copyright and privacy issues |
| `test_dataset/` and `test_dataset_insta/` (95 images) | Same; some screenshots contain identifiable people |
| `best_model/*.pth` (~32–35 MB each) | Binary weights, not code; can be regenerated; not suitable for a code portfolio |
| `test_dataset_insta.zip` | Duplicate of the test set |
| `Presentacion P3 VxC.pdf` | Course presentation with personal/university information; not code |
| `P3_INFORME.docx`, `VxC_Lab_session_4_v3 (1).pdf` | Course report/lab material; personal/university information; not code |
| `Google-Image-Scraper-master/` | Third-party tool; not written for this project |
| `guia_entrenamiento.html` | Internal training guide; overlaps with the docs and is partly tied to the course workflow |
| `Xception_Training.ipynb` | Exploratory notebook; not part of the final two-phase pipeline |
| `pipeline_transfer_learning.svg` | Outdated diagram (it says MobileNetV3-small, 5 + 3 epochs, blocks 4–5); replaced with accurate Mermaid diagrams |
| `P3.code-workspace`, `.vscode/`, `.claude/`, `.latex-tools/` | Editor/tooling configuration; not project code |
| Prediction-grid images from `resultados/` | They show a person's face (Instagram test screenshots); excluded for privacy |
| Duplicate training scripts | Consolidated into one CLI with `--model`; keeping three near-identical files would hurt readability |
| `Fotos/imagenet1000_clsidx_to_labels_list.txt` | ImageNet label list used for exploration; not needed for the six-class classifier |

The original folder remains untouched. This repository is a new, clean copy
under `CNN_for_VisualRecognition/`.

## Code improvements made for the portfolio version

The original scripts worked, but they had several issues that would be visible
in a code review. The portfolio version fixes them:

1. **Checkpointing bug.** The original `best_val_loss` comparison happened
   after the epoch loop, so the last epoch was always saved, not the best one.
2. **Phase-2 overwrite.** Phase 2 overwrote the phase-1 checkpoint
   unconditionally. The new version restores the best phase-1 weights before
   fine-tuning and keeps the best checkpoint across both phases.
3. **Random split → stratified split.** Every class is now represented in train
   and validation.
4. **Hard-coded Windows path removed.** `--output-dir` replaces the absolute
   path from the original script.
5. **BatchNorm side effect removed.** The original dummy forward pass ran in
   training mode, which could update BatchNorm running statistics. The new code
   uses `backbone.num_features`, with an eval-mode fallback.
6. **Correct metric aggregation.** Per-batch accuracy averaging was replaced by
   sample-weighted loss and accuracy over the whole split.
7. **Model-aware preprocessing.** The evaluation resize is derived from the
   backbone's input size instead of a hard-coded 256.
8. **Self-describing checkpoints.** Class names, input size, mean/std, dropout
   and config are stored with the weights, enabling standalone prediction and
   ONNX export.
9. **One CLI instead of three scripts.** `--model` selects MobileNetV3,
   EfficientNet-Lite0 or another timm backbone.
10. **Deployment path added.** ONNX export with embedded preprocessing metadata
    plus a PyTorch-free inference script.

These are engineering changes. The method (transfer learning, two-phase
fine-tuning, augmentation, evaluation) remains the original project's.

## Note on the reported accuracy

The original presentation reports **93.6%** for MobileNetV3-Large and **92.8%**
for EfficientNet-Lite0. The confusion matrices preserved in `resultados/`,
however, correspond to **90/95 = 94.7%** for both models.

This repository uses the **matrix-backed** numbers (94.7%, macro recall 93.9%
and 93.5%) because the matrices are the artifact that was preserved and can be
verified. The difference is consistent with run-to-run variance on a 95-image
test set: one image is 1.05 percentage points, and GPU training is not
bit-for-bit deterministic. The honest summary is **"roughly 93–95%"**, not a
single precise number.

## Licence summary

The repository uses a custom **Academic Portfolio License**:

- free to read, study, run and modify for personal learning, teaching and
  non-commercial academic research;
- attribution to the project **and** to the Computer Vision course/teaching
  team at EINA, Universidad de Zaragoza, is required;
- reuse in another project, redistribution, commercial use or publication of a
  derivative work requires prior permission from the author and the teaching
  team;
- the dataset images are not covered and are not distributed.

The README and `LICENSE` both give prominence to the professors' supervision
and guidance, as requested.

## Publishing

- Repository: <https://github.com/sejotta/CNN_for_VisualRecognition>
- Branch: `main`
- The repository contains only text/code and small result figures. No dataset,
  weights, PDFs, reports, personal images or third-party tools are committed.
- The local commit uses the GitHub handle and GitHub's noreply address, not a
  personal email.

## Resumen en español

- Subí solo el código limpio, la documentación, las figuras de resultados sin
  datos personales y la licencia. Dejé fuera datasets, pesos `.pth`, PDFs,
  informes, herramientas de terceros y capturas con personas.
- **No encontré código de CoppeliaSim ni de robot en P3.** El proyecto es un
  clasificador CNN de seis monumentos. Lo único relacionado con despliegue es
  el flujo ONNX / Raspberry Pi, que sí está incluido y documentado.
- La licencia permite usar el código para aprender, pero para reutilizarlo hay
  que pedir permiso y dar crédito a los profesores de Computer Vision de la
  EINA (Universidad de Zaragoza).
- Los resultados publicados usan las matrices de confusión conservadas:
  90/95 (94.7%) para MobileNetV3-Large y EfficientNet-Lite0. La presentación
  original redondeaba a 93.6% y 92.8%; la diferencia entra dentro de la
  variación normal de un test de 95 imágenes.
