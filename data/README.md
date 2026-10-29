# Dataset

The dataset is **not included** in this repository. The images are third-party
web content, the full training folder is about 440 MB, and some test images
contain people. Redistributing them would raise copyright and privacy issues.

## Expected layout

`torchvision.datasets.ImageFolder` expects one folder per class:

```text
data/
├── train/
│   ├── Brooklyn_Bridge/
│   │   ├── image_0001.jpg
│   │   └── ...
│   ├── Eiffel_Tower/
│   ├── Golden_Gate/
│   ├── Great_Wall/
│   ├── Sagrada_Familia/
│   └── Tokio_Tower/
└── test/
    ├── Brooklyn_Bridge/
    ├── Eiffel_Tower/
    ├── Golden_Gate/
    ├── Great_Wall/
    ├── Sagrada_Familia/
    └── Tokio_Tower/
```

Class names are read from the folder names and sorted alphabetically. The
validation split is created **inside the code** from `data/train` (stratified,
75/25 by default), so do not create a separate `val/` folder.

## Original dataset used for the reported results

| Class | Train images | Test images |
|---|---:|---:|
| Brooklyn Bridge | 485 | 17 |
| Eiffel Tower | 689 | 22 |
| Golden Gate | 549 | 12 |
| Great Wall | 646 | 17 |
| Sagrada Família | 357 | 15 |
| Tokyo Tower (`Tokio_Tower`) | 314 | 12 |
| **Total** | **3,040** | **95** |

The reported test set is the 95-image **Instagram-style** set: screenshots with
people, text overlays, filters, crops and unusual viewpoints. An earlier,
easier 30-image test set was used during development but is not the set behind
the final results.

## Why this split matters

The training images were collected to cover different seasons, times of day,
perspectives, zooms, colour filters, black-and-white photos and even paintings
or AI-generated images. The test set was kept separate and intentionally looks
different from the training distribution. That domain shift is the main reason
the test accuracy is lower than the validation accuracy.

## Using your own data

Any set of classes works, as long as `data/train` and `data/test` contain the
same class folder names. For a small dataset, transfer learning is still
effective, but aim for at least 100–200 images per class. The code validates
that every class has at least two training images (required for a stratified
split) and that train/test class folders match.

## Why the images are not committed

- **Size:** roughly 440 MB for the training set alone.
- **Copyright:** the images come from public web sources and remain subject to
  their original licences. The repository's licence does not cover them.
- **Privacy:** some Instagram-style test screenshots include identifiable
  people, so those images must not be published.

The model weights are also excluded (they are binary artifacts and can be
regenerated with `python -m landmark_classifier.train`).
