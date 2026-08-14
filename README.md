# Wheat Crop Segmentation Comparison

This repository contains a compact computer vision benchmark for segmenting wheat crops from field images. It compares three families of methods:

- classical segmentation: Watershed, MSER, and SLIC superpixels;
- machine learning with handcrafted features: HOG, LBP, GLCM, and their combinations with SVM, Random Forest, and KNN classifiers;
- deep learning: U-Net, Attention U-Net, and DeepLabV3+.

The project is intended as a readable code portfolio artifact: the data, trained weights, and generated figures are intentionally excluded, while the model definitions, experiment pipeline, notebooks, and reproduction instructions are kept in the repository.

## Repository Layout

```text
wheat-crop-segmentation-comparison/
|-- main.py
|-- experiment_config.py
|-- experiment_runner.py
|-- helpers/
|   `-- helper_fcts.py
|-- methods/
|   |-- base.py
|   |-- unet.py
|   |-- attention_unet.py
|   |-- dlv3plus.py
|   `-- dl_common.py
|-- notebooks/
|   |-- classical_segmentation_ml.ipynb
|   `-- deep_learning_comparison.ipynb
|-- utils/
|   `-- visualization.py
|-- data/
|   `-- README.md
|-- requirements.txt
`-- .gitignore
```

## What This Shows

The code demonstrates an end-to-end segmentation experiment workflow:

- loading paired field images and binary plant/soil masks;
- applying image preprocessing and data augmentation;
- training multiple segmentation models under a shared protocol;
- computing precision, recall, F1-score, IoU, and accuracy;
- exporting metric tables, training histories, and qualitative prediction samples;
- running ablation studies for pretraining, augmentation, and attention modules.

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches your CUDA version first. For example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Data

The original image datasets are not included. Place data locally using this structure:

```text
data/
|-- EWS-Dataset/
|   |-- train/
|   |-- validation/
|   `-- test/
`-- DataforYuetal/
    |-- train/
    |   |-- HLC/
    |   `-- LLC/
    |-- validation/
    |   |-- HLC/
    |   `-- LLC/
    `-- test/
        |-- HLC/
        `-- LLC/
```

For the deep learning pipeline, the expected EWS split can also be placed directly under:

```text
data/train/
data/validation/
data/test/
```

Each image should have a matching binary mask named with the `_mask.png` suffix, for example:

```text
sample_6.png
sample_6_mask.png
```

## Run The Deep Learning Pipeline

From the repository root:

```bash
python main.py
```

This trains and evaluates the configured U-Net, Attention U-Net, and DeepLabV3+ experiments. Outputs are written to `results/`, which is ignored by Git.

## Notebooks

Launch Jupyter from the repository root so the relative paths resolve correctly:

```bash
jupyter lab
```

Then open:

- `notebooks/classical_segmentation_ml.ipynb` for Watershed/MSER/Superpixel and handcrafted-feature ML experiments;
- `notebooks/deep_learning_comparison.ipynb` for the deep learning model comparison and ablation workflow.

Notebook outputs are cleared so the repository stays lightweight.

## External Sources

This repository uses standard scientific Python libraries and `segmentation-models-pytorch` for the library-backed U-Net and DeepLabV3+ implementations.

Some dataset loading and benchmarking utilities were adapted from the EWS benchmark code associated with:

Zenkl et al., "Outdoor Plant Segmentation With Deep Learning for High-Throughput Field Phenotyping on a Diverse Wheat Dataset", Frontiers in Plant Science, 2022.

The custom Attention U-Net implementation, experiment orchestration, comparison tables, ablation setup, and visualization utilities are organized in this repository for reproducible model comparison.
