# Pet Breed Classification and Segmentation

This project implements both classification and semantic segmentation models for the Oxford-IIIT Pet Dataset, providing tools for breed identification and pet outline segmentation.

## Features

- Multi-class classification of 37 pet breeds
- Semantic segmentation for pet outlines
- Comprehensive evaluation metrics
- Detailed visualizations

## Installation

```bash
git clone https://github.com/engyyezzatt/Oxford_IIIT_Pet_classification_Seg.git
cd Oxford_IIIT_Pet_classification_Seg
pip install -r requirements.txt
```

## Project Structure

```
project/
├── src/            # Source code
├── tests/          # Unit tests
├── notebooks/      # Jupyter notebooks
├── docs/           # Documentation
└── data/           # Dataset directory
```

## Usage

### Training the Classification Model

```python
from src.data.preprocessor import PetDatasetPreprocessor
from src.models.classifier import PetClassifier

# Initialize data and model
preprocessor = PetDatasetPreprocessor("./data")
dataloaders = preprocessor.load_data()
model = PetClassifier(num_classes=37)

# Train the model
model.train(dataloaders['train'], dataloaders['val'], epochs=10)
```

### Training the Segmentation Model

```python
from src.models.segmentation import SegmentationTrainer

trainer = SegmentationTrainer()
trainer.train_epoch(dataloaders['train'])
```

## Evaluation Metrics

### Classification
- Accuracy
- Loss
- Confusion Matrix

### Segmentation
- Intersection over Union (IoU)
- Dice Coefficient
- Pixel Accuracy

## Results

- Segmentation evaluation plots: 
![Segmentation evaluation plots](plots/Seg_evaluation_plots.png)

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
