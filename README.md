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
└── models/           # Saved models directory
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

# Results

## Classification evaluation plots 

### Fine-Tuned-ResNet34

1. Train-Val Accuracy Plot 
![acc plot](plots/Resnet_classification_acc.png)


2. Train-Val Loss Plot 
![loss plot](plots/Resnet_Classification_loss.png)


3. Confusion Matrix 
![Confusion Matrix plot](plots/Resnet_classification_conf.png)


### Simple CNN Model 
1. Train-Val Accuracy&Loss Plot 
![Accuracy&Loss plot](plots/CNN_classification_acc_loss.png)


2. Confusion Matrix 
![Confusion Matrix plot](plots/CNN_classification_conf.png)


## Segmentation evaluation plots: 

1. IOU & Dice Coefficient & Pixel Accuracy
![evaluation plots](plots/Seg_evaluation_plots.png)



## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
