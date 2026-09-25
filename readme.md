# Solar Panel Defect Classification Using Deep Learning

A two-stage image classification system for detecting and identifying defects on solar panels, built with TensorFlow/Keras and served via FastAPI.

## Overview

This project classifies solar panel images in two stages:
1. **Binary classification** — clean vs. defective
2. **Multi-class classification** (optional) — identifies the specific defect type: Bird-drop, Dusty, Electrical-damage, Physical-Damage, or Snow-Covered

Both models use transfer learning with a MobileNetV3Small backbone (pretrained on ImageNet), fine-tuned on a solar panel defect dataset.

## Tech Stack

- **Modeling**: TensorFlow/Keras, MobileNetV3Small (transfer learning)
- **API**: FastAPI, Uvicorn
- **Data handling**: scikit-learn, NumPy, Pillow, OpenCV
- **Deployment**: Docker

## Project Structure
├── data/ # Dataset (not included in repo — see Setup)
├── src/
│ ├── data_prep.py # Splits raw dataset into train/val/test
│ ├── dataset.py # tf.data pipeline (loading, augmentation)
│ ├── model_binary.py # Binary classifier architecture
│ ├── model_multiclass.py # Multi-class classifier architecture
│ ├── train.py # Trains the binary model
│ ├── train_multiclass.py # Trains the multi-class model
│ ├── evaluate.py # Binary model evaluation (confusion matrix, threshold tuning)
│ ├── evaluate_multiclass.py # Multi-class model evaluation
│ └── predict.py # End-to-end inference (both stages)
├── api/
│ ├── main.py # FastAPI app
│ └── schemas.py # Request/response models
├── models/ # Trained model files (.keras)
├── deployment/
│ └── Dockerfile
└── requirements.txt


## Setup

1. Clone the repo and install dependencies:
```bash
   pip install -r requirements.txt
```

2. Download the dataset from Kaggle ([pythonafroz/solar-panel-images](https://www.kaggle.com/datasets/pythonafroz/solar-panel-images)) into `data/raw/faulty_solar_panel/`

3. Prepare the data splits:
```bash
   python src/data_prep.py
```

4. Train the models:
```bash
   python src/train.py
   python src/train_multiclass.py
```

5. Run the API locally:
```bash
   uvicorn api.main:app --reload
```
   Visit `http://127.0.0.1:8000/docs` for the interactive API docs.

## Results

| Model | Test Accuracy | Notes |
|---|---|---|
| Binary (clean/defective) | 84% (at threshold 0.2) | Threshold tuned via precision-recall curve for higher defect recall |
| Multi-class (defect type) | 58% | Bird-drop class shows lower recall due to visual similarity with Dusty defects |

## Key Design Decisions

- **Threshold tuning**: The binary model's default 0.5 threshold favored precision over recall. Since missing a real defect is costlier than a false alarm in an inspection context, the threshold was lowered to 0.2 (F1-optimal), improving defect recall from 57% to 96%.
- **Two-phase training**: Each model trains in two phases — feature extraction (frozen backbone) followed by fine-tuning (partial backbone unfreezing at a low learning rate).