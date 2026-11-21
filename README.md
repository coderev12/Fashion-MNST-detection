# Fashion MNIST Classification using Machine Learning and CNN

## Project Overview

This project focuses on classifying clothing images into 10 categories using the Fashion MNIST dataset.

It explores multiple machine learning algorithms, starting with traditional classification models and progressing to Deep Learning using Convolutional Neural Networks (CNN).

The objective is to compare different approaches, understand their performance, and identify the most suitable model for image classification.

**Final Result: CNN achieved 91.93% accuracy on the unseen test dataset.**

## Dataset

Fashion MNIST contains 70,000 grayscale clothing images, each of size 28 × 28 pixels.

| Dataset | Images |
|---|---:|
| Training | 60,000 |
| Test | 10,000 |

The training dataset was divided into:

- Training: 54,000 images
- Validation: 6,000 images

**10 Clothing Categories:**

T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.

## Project Workflow

1. Dataset loading and exploratory data analysis.
2. Image visualization and class distribution analysis.
3. Data preprocessing and pixel normalization (0–1).
4. Training and evaluation of traditional ML models.
5. Hyperparameter tuning and model comparison.
6. CNN architecture development and training.
7. Overfitting analysis and EarlyStopping.
8. Confusion matrix and classification report analysis.
9. CNN feature map and filter visualization.
10. Final evaluation on unseen test data.

## Machine Learning Models

Three traditional machine learning algorithms were implemented and evaluated.

### 1. Logistic Regression

Used as a baseline classification model to understand how traditional machine learning performs on image data.

- Training Accuracy: 86.59%
- Validation Accuracy: 85.53%

### 2. Decision Tree

A tree-based classification model. Different maximum depths were tested to understand model complexity and overfitting.

- Best Maximum Depth: 15
- Training Accuracy: 94.34%
- Validation Accuracy: 80.62%

### 3. Random Forest

An ensemble learning algorithm combining multiple Decision Trees. Hyperparameters such as `n_estimators`, `max_depth`, `min_samples_leaf`, and `min_samples_split` were tuned.

**Best Configuration:**

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=2,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
```

- Training Accuracy: 98.79%
- Validation Accuracy: 87.87%

## Why CNN?

Traditional machine learning models require images to be flattened into one-dimensional feature vectors, which removes their explicit spatial structure.

CNNs are specifically designed for image processing and offer several advantages:

- Preserve spatial relationships between pixels.
- Automatically extract visual features.
- Detect patterns such as edges, shapes, and textures.
- Learn increasingly complex representations through deeper layers.
- Use shared convolution filters to reduce parameter requirements.

CNN achieved better validation accuracy than all three traditional models, making it the most suitable approach for this classification task.

## CNN Architecture

The CNN was implemented using TensorFlow and Keras.

```python
cnn_model = Sequential([
    Input(shape=(28, 28, 1)),

    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.3),

    Dense(10, activation='softmax')
])
```

**Architecture Details:**

| Layer | Purpose |
|---|---|
| Conv2D (32 filters) | Extracts basic visual features |
| MaxPooling2D | Reduces spatial dimensions |
| Conv2D (64 filters) | Learns complex feature representations |
| MaxPooling2D | Further reduces feature dimensions |
| Flatten | Converts feature maps into a vector |
| Dense (128) | Learns classification patterns |
| Dropout (0.3) | Helps prevent overfitting |
| Dense (10) | Produces probabilities for 10 classes |

**Total Parameters:** 225,034

### Training Configuration

- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Batch Size: 64
- Maximum Epochs: 20 (EarlyStopping)
- EarlyStopping: Monitored validation loss with patience 3 and best-weight restoration.

## Model Evaluation

The CNN was evaluated using accuracy, precision, recall, F1-score, and a confusion matrix.

### Validation Classification Report

| Category | Precision | Recall | F1-score |
|---|---:|---:|---:|
| T-shirt/top | 0.85 | 0.89 | 0.87 |
| Trouser | 0.99 | 0.99 | 0.99 |
| Pullover | 0.88 | 0.88 | 0.88 |
| Dress | 0.95 | 0.89 | 0.92 |
| Coat | 0.88 | 0.85 | 0.87 |
| Sandal | 0.99 | 0.97 | 0.98 |
| Shirt | 0.76 | 0.78 | 0.77 |
| Sneaker | 0.94 | 0.97 | 0.95 |
| Bag | 0.97 | 0.98 | 0.98 |
| Ankle boot | 0.97 | 0.96 | 0.97 |

**Key Observation:** Shirt was the most challenging category, frequently being confused with T-shirt/top, Pullover, and Coat.

## CNN Experiments

### 1. Feature Map Visualization

Extracted and visualized feature maps from both convolution layers to understand how the CNN learns visual representations.

- First convolution layer: 32 feature maps.
- Second convolution layer: 64 feature maps.

### 2. Filter Weight Visualization

Visualized the learned 3 × 3 convolution filters to understand the weights automatically learned during training.

### 3. CNN Without MaxPooling

Experimented with removing MaxPooling layers to understand their impact on performance.

| Metric | With MaxPooling | Without MaxPooling |
|---|---:|---:|
| Best Validation Accuracy | 91.75% | 91.72% |
| Flattened Features | 1,600 | 36,864 |

Removing MaxPooling increased the model size and computational requirements without improving validation accuracy meaningfully.

## Final Model Comparison

| Model | Validation Accuracy |
|---|---:|
| Logistic Regression | 85.53% |
| Decision Tree | 80.62% |
| Random Forest | 87.87% |
| **CNN** | **91.75%** |

### Final Test Results

The selected CNN was evaluated on the separate 10,000-image test dataset.

| Metric | Result |
|---|---:|
| Test Accuracy | **91.93%** |
| Test Loss | **0.2479** |
| Correct Predictions | Approximately 9,193 |
| Incorrect Predictions | Approximately 807 |

CNN outperformed the traditional machine learning models and demonstrated good generalization on unseen images.

## Technologies Used

- **Python** – Programming language
- **NumPy & Pandas** – Data manipulation
- **Matplotlib** – Data and feature visualization
- **Scikit-learn** – Traditional ML models and evaluation
- **TensorFlow & Keras** – CNN development and training
- **Google Colab / Jupyter Notebook** – Development environment

## Key Learnings

- Data preprocessing and normalization.
- Training, validation, and test data separation.
- Traditional machine learning algorithms and ensemble learning.
- Hyperparameter tuning and overfitting.
- CNN architecture and convolution operations.
- Feature extraction and filter visualization.
- Model evaluation using classification metrics.
- Understanding the importance of model architecture in image classification.

## Conclusion

This project demonstrates the progression from traditional machine learning algorithms to deep learning for image classification.

Through experimentation and evaluation, CNN proved to be the most effective model for the Fashion MNIST dataset, achieving **91.93% test accuracy**.

The project provided practical experience in developing, optimizing, evaluating, and understanding image classification models using machine learning and deep learning techniques.
