# Brain Tumor MRI Classification + Explainable AI

An EfficientNet-B0 classifier for brain tumor MRI scans (glioma, meningioma, pituitary, no tumor),
paired with two complementary explainability methods — Grad-CAM and Integrated Gradients — to make
the model's predictions inspectable rather than a black box.

## Problem

Automated tumor classification from MRI is only useful in practice if clinicians (or downstream
reviewers) can verify *why* the model reached a decision, not just *what* it decided. This project
treats explainability as a first-class part of the pipeline, not an afterthought: every prediction
can be inspected through two independent attribution methods that catch different kinds of model
behavior.

## Architecture

```
MRI scan
   │
   ▼
Contour-based crop (OpenCV) — isolates brain region, discards scan margin/background
   │
   ▼
EfficientNet-B0 (ImageNet-pretrained, fine-tuned head)
   │
   ├─► Classification: glioma / meningioma / pituitary / notumor
   │
   ├─► Grad-CAM        — hooks the final conv block, shows which spatial regions
   │                      of the feature map drove the prediction (coarse, activation-based)
   │
   └─► Integrated Gradients (Captum) — attributes the prediction back to individual
                                         input pixels via gradient integration from a
                                         black-image baseline (fine-grained, gradient-based)
```

## Why two XAI methods, not one

Grad-CAM and Integrated Gradients answer different questions and have different failure modes:

| | Grad-CAM | Integrated Gradients |
|---|---|---|
| Basis | Activations of a late conv layer | Gradients w.r.t. input pixels |
| Resolution | Coarse (bounded by feature map size) | Pixel-level |
| Reads as | Region-level "where did it look" | Pixel-level "what exactly mattered" |
| Weakness | Can't explain fine detail | Can be visually noisy/speckled |

Using both gives a sanity check on each other: if Grad-CAM's region and IG's pixel cluster agree,
that's stronger evidence the model is keying off the actual lesion rather than an artifact.

## Key finding: a shortcut-learning signal

Running both methods across a wider test sample surfaced something worth flagging rather than
hiding: on some correctly-classified `notumor` scans, Grad-CAM attention concentrated on the scan's
background/corner region rather than brain tissue. The model reached the right label, but the
attribution suggests it may be partly relying on an incidental cue (e.g. absence of a bright lesion
signature near the scan edge) rather than confirming healthy tissue directly. This is exactly the
kind of failure mode that accuracy metrics alone would never reveal — it only shows up because the
pipeline includes explainability by design.

See `saved_model/xai_comparison_grid.png` for the comparison grid these examples were drawn from,
including one misclassified case (meningioma predicted as glioma) where Grad-CAM/IG both focus
tightly on the correct lesion despite the wrong class label — evidence the model localizes correctly
even when the fine-grained class boundary is wrong.

## Evaluation

<!-- TODO: paste your classification_report output here -->
| Class | Precision | Recall | F1 |
|---|---|---|---|
| glioma | _fill in_ | _fill in_ | _fill in_ |
| meningioma | _fill in_ | _fill in_ | _fill in_ |
| notumor | _fill in_ | _fill in_ | _fill in_ |
| pituitary | _fill in_ | _fill in_ | _fill in_ |
| **Overall accuracy** | | | _fill in_ |

Full confusion matrix: `saved_model/confusion_matrix.png`

## Repo structure

```
.
├── brain_tumor_xai.ipynb          # full pipeline: data → train → eval → Grad-CAM → IG → comparison grid
├── saved_model/
│   ├── efficientnet_b0_brain_tumor.pt
│   ├── confusion_matrix.png
│   └── xai_comparison_grid.png
├── requirements.txt
└── README.md
```

## How to run

1. Get a Kaggle API token (kaggle.com → Account → Create New Token) for the automatic dataset download,
   or manually download [brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).
2. Open `brain_tumor_xai.ipynb` in Google Colab (GPU runtime recommended) or a local Jupyter environment.
3. Install dependencies: `pip install -r requirements.txt`
4. Run all cells top to bottom. Training takes ~10-15 minutes on a Colab GPU.
5. The trained checkpoint, confusion matrix, and XAI comparison grid are saved to `saved_model/`.

## Tech stack

PyTorch, torchvision (EfficientNet-B0), OpenCV, Captum (Integrated Gradients), scikit-learn (evaluation),
opendatasets (Kaggle download).

## Author

Bhavnish Singhall — M.Sc. Computational Linguistics, University of Stuttgart
