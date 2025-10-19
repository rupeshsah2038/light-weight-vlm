# light-weight-vlm-with-KD
A lightweight vision-language model (VLM) training framework for the MedPix 2.0 dataset. This script uses **knowledge distillation** to compress a large teacher model (CLIP + DistilBERT) into a smaller student model under 1B parameters, optimized for both **modality** and **anatomical location** classification.

## Features

* Combines **visual and textual** understanding.
* Teacher-student training via **knowledge distillation**.
* Built-in **caching**, **quantization**, and **pruning** for efficient inference.
* Logs metrics, confusion matrices, and validation results.

## Requirements

Python 3.8+

Install dependencies:

```bash
pip install torch torchvision transformers jsonlines scikit-learn pillow numpy
```

## Dataset Structure

The code expects the MedPix 2.0 dataset organized as follows:

```
MedPix-2-0/
├── images/
│   ├── <image files>.png
├── splitted_dataset/
│   ├── data_train.jsonl
│   ├── data_dev.jsonl
│   ├── descriptions_train.jsonl
│   ├── descriptions_dev.jsonl
```

## How It Works

1. **Teacher model:** Uses pretrained CLIP and DistilBERT models for supervision.
2. **Student model:** A custom lightweight CNN + LSTM-based VLM.
3. **Distillation loss:** Combines cross-entropy and KL-divergence for soft label learning.
4. **Optimization:** Includes mixed precision training, layer freezing/unfreezing, pruning, and dynamic quantization.

## Training

Run the main script:

```bash
python light-weight-vlm.py
```

Training logs and results will be saved in the following directories:

```
logs/
  ├── training_log.csv
  ├── confusion_modality_epoch_*.csv
  ├── confusion_location_epoch_*.csv
results/
  ├── lightweight_vlm.pth
```

## Outputs

* `lightweight_vlm.pth`: The trained lightweight model.
* `training_log.csv`: Per-epoch loss and accuracy metrics.
* Confusion matrices for both modality and location predictions.

## Notes

* The dataset paths may need to be adjusted to your local setup.
* GPU acceleration is recommended.
* Quantized model is stored for efficient deployment.

## License

This project is provided for research and educational use only.
