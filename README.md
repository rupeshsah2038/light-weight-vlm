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
Running logs:
**Teacher Model parameters: 217650441
Student Model parameters: 221888136**
Epoch 1: Loss: 442.3589, Modality Acc: 0.6497, Location Acc: 0.2995
Epoch 2: Loss: 390.7010, Modality Acc: 0.6294, Location Acc: 0.0914
Epoch 3: Loss: 153.5840, Modality Acc: 0.4822, Location Acc: 0.1269
Epoch 4: Loss: 80.9659, Modality Acc: 0.7614, Location Acc: 0.2030
Epoch 5: Loss: 38.4006, Modality Acc: 0.4569, Location Acc: 0.1878
Epoch 6: Loss: 8.9546, Modality Acc: 0.5736, Location Acc: 0.3553
Epoch 7: Loss: 1.1095, Modality Acc: 0.8528, Location Acc: 0.7563
Epoch 8: Loss: 0.9501, Modality Acc: 0.9239, Location Acc: 0.7970
Epoch 9: Loss: 0.8788, Modality Acc: 0.9746, Location Acc: 0.9239
Epoch 10: Loss: 0.8472, Modality Acc: 0.9746, Location Acc: 0.9492
**Optimized Student Model parameters: 15765888**

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
