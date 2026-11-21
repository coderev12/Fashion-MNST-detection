# Fashion MNIST Classification using Machine Learning and CNN

## Project Overview

This project focuses on building and evaluating machine learning models to classify images of clothing into 10 different categories using the **Fashion MNIST dataset**.

The project explores multiple machine learning algorithms, starting with traditional classification techniques and gradually progressing towards Deep Learning using Convolutional Neural Networks (CNNs).

The primary objective is not only to achieve good classification accuracy but also to understand how different models learn image features, how their performance can be improved, and why CNNs are particularly effective for image classification tasks.

After experimenting with Logistic Regression, Decision Tree, Random Forest, and CNN, the CNN achieved the best overall performance, reaching **91.93% accuracy on the unseen test dataset**.

---

## Table of Contents

1. [Dataset Description](#1-dataset-description)
2. [Project Workflow](#2-project-workflow)
3. [Data Preprocessing](#3-data-preprocessing)
4. [Exploratory Data Analysis](#4-exploratory-data-analysis)
5. [Machine Learning Models](#5-machine-learning-models)
   - Logistic Regression
   - Decision Tree
   - Random Forest
6. [Why CNN?](#6-why-cnn)
7. [CNN Architecture](#7-cnn-architecture)
8. [CNN Training and Optimization](#8-cnn-training-and-optimization)
9. [Model Evaluation](#9-model-evaluation)
10. [CNN Feature Map Visualization](#10-cnn-feature-map-visualization)
11. [Filter Weight Visualization](#11-filter-weight-visualization)
12. [Experiment: CNN Without MaxPooling](#12-experiment-cnn-without-maxpooling)
13. [Final Model Comparison](#13-final-model-comparison)
14. [Key Learnings](#14-key-learnings)
15. [Technologies Used](#15-technologies-used)
16. [How to Run the Project](#16-how-to-run-the-project)
17. [Conclusion](#17-conclusion)

---

## 1. Dataset Description

The project uses the **Fashion MNIST dataset**, a collection of grayscale images representing different clothing and footwear categories.

Each image is:

- 28 × 28 pixels.
- Grayscale (single channel).
- Associated with one of 10 clothing categories.

### Dataset Distribution

| Dataset | Number of Images |
|---|---:|
| Training dataset | 60,000 |
| Test dataset | 10,000 |
| Total | 70,000 |

The original training dataset was further divided into training and validation datasets.

| Split | Number of Images |
|---|---:|
| Training | 54,000 |
| Validation | 6,000 |
| Test | 10,000 |

### Classes

The dataset contains the following 10 categories:

| Label | Category |
|---:|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

Each category contains 6,000 training images and 1,000 test images.

---

## 2. Project Workflow

The project follows a systematic machine learning workflow.

1. Load and understand the Fashion MNIST dataset.
2. Perform exploratory data analysis.
3. Visualize sample clothing images.
4. Normalize pixel values.
5. Split the training dataset into training and validation sets.
6. Flatten images for traditional machine learning models.
7. Train Logistic Regression.
8. Train and optimize a Decision Tree.
9. Train and optimize a Random Forest.
10. Compare traditional machine learning models.
11. Prepare image data for CNN.
12. Design and train a CNN architecture.
13. Apply EarlyStopping to control overfitting.
14. Evaluate the CNN using accuracy, precision, recall, F1-score, and confusion matrix.
15. Visualize convolutional feature maps and learned filter weights.
16. Experiment with removing MaxPooling layers.
17. Evaluate the selected CNN on the unseen test dataset.

---

## 3. Data Preprocessing

### 3.1 Loading the Dataset

The image features and labels were separated from the dataset.

- Features contain pixel values.
- Labels contain the corresponding clothing category (0–9).

### 3.2 Image Reshaping

Each image contains 784 pixel values.

For CNN, these values were reshaped into:

`28 × 28 × 1`

The final dimension represents the grayscale channel.

### 3.3 Normalization

Original pixel values range from 0 to 255.

To make the input values consistent and easier for neural networks to process, pixel values were normalized:

\[
X_{normalized} = \frac{X}{255}
\]

After normalization, pixel values range from 0 to 1.

### 3.4 Training and Validation Split

The original 60,000 training images were divided into:

- 54,000 training images.
- 6,000 validation images.

The split used stratified sampling to preserve the distribution of all 10 categories.

```python
X_train, X_val, y_train, y_val = train_test_split(
    train_images,
    label,
    test_size=0.1,
    random_state=42,
    stratify=label
)
```

The separate 10,000-image test dataset was kept untouched during training and model selection.

### 3.5 Flattening for Traditional ML Models

Traditional machine learning algorithms expect a two-dimensional feature matrix.

Therefore, each 28 × 28 image was converted into a 784-dimensional feature vector.

```python
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
```

Resulting shapes:

- Training: (54000, 784)
- Validation: (6000, 784)

CNN models, on the other hand, used the original two-dimensional image structure.

---

## 4. Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed to understand the dataset before training models.

The analysis included:

- Checking dataset dimensions.
- Understanding class distribution.
- Visualizing individual clothing images.
- Displaying multiple images in a grid.
- Examining pixel intensity values.
- Understanding image dimensions and grayscale representation.

This helped establish the nature of the dataset and identify the preprocessing requirements for different machine learning algorithms.

---

## 5. Machine Learning Models

Before implementing CNN, three traditional machine learning algorithms were trained and evaluated.

### 5.1 Logistic Regression

**What is it?**

Logistic Regression is a supervised classification algorithm that predicts the probability of an input belonging to different classes.

Although its name contains regression, it is widely used for classification problems.

**Implementation:**

```python
model = LogisticRegression(
    max_iter=1000,
    tol=1e-3,
    solver='lbfgs'
)

model.fit(X_train_flat, y_train)
```

**Why was it used?**

- Provides a baseline classification model.
- Helps understand how traditional ML algorithms perform on image data.
- Offers a useful comparison against more complex models.

**Results:**

| Metric | Accuracy |
|---|---:|
| Training Accuracy | 86.59% |
| Validation Accuracy | 85.53% |

**Observation:**

Logistic Regression provided a reasonable baseline but struggled with visually similar categories, especially Shirt, Pullover, and T-shirt/top.

Since it operates on flattened pixel values, it does not explicitly learn spatial relationships between neighboring pixels.

### 5.2 Decision Tree Classifier

**What is it?**

A Decision Tree is a supervised learning algorithm that makes predictions through a sequence of decisions based on input features.

It divides the feature space into smaller regions using learned decision rules.

**Implementation:**

```python
DecisionTreeClassifier(random_state=42)
```

Different maximum depths were tested to understand the effect of model complexity.

**Hyperparameter Experimentation:**

| Maximum Depth | Training Accuracy | Validation Accuracy |
|---:|---:|---:|
| 5 | 71.04% | 69.78% |
| 10 | 85.14% | 80.17% |
| 15 | 94.34% | 80.62% |
| 20 | 98.65% | 79.38% |
| 25 | 99.52% | 78.95% |
| 30 | 99.84% | 79.08% |

**Observation:**

As the tree depth increased, training accuracy improved significantly, but validation accuracy stopped improving.

This demonstrated overfitting.

The model started memorizing training data instead of learning patterns that generalized well to unseen images.

The best validation accuracy among the tested depths was **80.62% at maximum depth 15**.

### 5.3 Random Forest Classifier

**What is it?**

Random Forest is an ensemble learning algorithm that combines predictions from multiple Decision Trees.

Instead of relying on one tree, it trains multiple trees and combines their predictions to improve stability and generalization.

**Why was it used?**

- Reduces dependence on a single Decision Tree.
- Helps control overfitting.
- Can capture nonlinear relationships.
- Provides a stronger traditional ML comparison.

**Hyperparameter Tuning:**

Different combinations of the following parameters were explored:

- `n_estimators`: Number of decision trees.
- `max_depth`: Maximum depth of each tree.
- `min_samples_leaf`: Minimum number of samples required at a leaf.
- `min_samples_split`: Minimum samples required to split a node.

**Best Configuration:**

```python
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=2,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
```

**Results:**

| Configuration | Training Accuracy | Validation Accuracy |
|---|---:|---:|
| 100 Trees | 99.84% | 79.08% |
| Max Depth = 20 | 99.37% | 87.28% |
| 200 Trees, Depth = 20 | 99.42% | 87.47% |
| Min Samples Leaf = 2 | 99.00% | 87.75% |
| Min Samples Split = 5 | 98.79% | **87.87%** |

**Observation:**

Random Forest significantly outperformed the Decision Tree after hyperparameter tuning.

However, it still struggled with visually similar clothing categories, particularly Shirt.

---

## 6. Why CNN?

After experimenting with traditional machine learning algorithms, Convolutional Neural Networks were introduced.

The main limitation of the earlier models was their treatment of images as flattened collections of pixel values.

For example, a 28 × 28 image becomes a one-dimensional vector containing 784 values.

Although these values contain image information, flattening removes the explicit two-dimensional spatial structure.

A CNN is specifically designed to process image data.

### Advantages of CNN

**1. Spatial Feature Learning**

CNNs preserve the spatial structure of images and learn relationships between neighboring pixels.

**2. Automatic Feature Extraction**

Instead of manually defining image features, CNNs learn useful patterns directly from training data.

**3. Convolution Operations**

Convolution filters detect visual patterns such as edges, boundaries, curves, and textures.

**4. Hierarchical Feature Learning**

Deeper convolution layers combine simpler features into more complex representations.

**5. Parameter Sharing**

The same filter is applied across different image locations, reducing the number of parameters compared with fully connected image-processing approaches.

### Why CNN was selected

The comparison showed that traditional ML models could achieve reasonable accuracy, but CNNs were better suited to learning visual patterns from clothing images.

CNN also achieved significantly better validation performance than the three traditional algorithms.

Therefore, CNN was selected as the primary model for this image classification task.

---

## 7. CNN Architecture

The CNN architecture was implemented using TensorFlow and Keras.

### Architecture

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

### Layer-by-Layer Explanation

| Layer | Configuration | Purpose |
|---|---|---|
| Input | 28 × 28 × 1 | Accepts grayscale images |
| Conv2D | 32 filters, 3 × 3 | Extracts basic visual patterns |
| MaxPooling2D | 2 × 2 | Reduces spatial dimensions |
| Conv2D | 64 filters, 3 × 3 | Learns more complex representations |
| MaxPooling2D | 2 × 2 | Further reduces spatial dimensions |
| Flatten | - | Converts feature maps into a vector |
| Dense | 128 neurons | Learns classification-related combinations |
| Dropout | 0.3 | Helps reduce overfitting |
| Output | 10 neurons | Predicts one of 10 clothing categories |

### Understanding Important Layers

**Convolution Layer**

Applies learnable filters to images to detect useful visual patterns.

**ReLU Activation**

Introduces non-linearity by replacing negative activations with zero.

\[
ReLU(x) = \max(0,x)
\]

**MaxPooling**

Selects the maximum activation from each 2 × 2 region, reducing spatial dimensions while retaining strong responses.

**Flatten**

Converts the multidimensional feature maps into a one-dimensional vector.

**Dense Layer**

Combines extracted features to learn classification patterns.

**Dropout**

Randomly disables 30% of neurons during training to help prevent overfitting.

**Softmax**

Converts the final output into probabilities across the 10 categories.

### Model Summary

| Metric | Value |
|---|---:|
| Total Parameters | 225,034 |
| Trainable Parameters | 225,034 |
| Non-trainable Parameters | 0 |

---

## 8. CNN Training and Optimization

### 8.1 Model Compilation

The CNN was compiled using the Adam optimizer and Sparse Categorical Crossentropy loss.

```python
cnn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

**Why these choices?**

- Adam: Efficiently updates model weights during training.
- Sparse Categorical Crossentropy: Suitable for multiclass classification when labels are integer values (0–9).
- Accuracy: Measures the percentage of correctly classified images.

### 8.2 Initial Training

The initial CNN was trained for 10 epochs using a batch size of 64.

```python
history = cnn_model.fit(
    X_train_cnn,
    y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_val_cnn, y_val)
)
```

**Observed Performance:**

| Metric | Result |
|---|---:|
| Training Accuracy at Epoch 10 | 96.21% |
| Best Validation Accuracy | 92.17% |
| Best Validation Epochs | 5 and 6 |

The training accuracy continued increasing, while validation accuracy began fluctuating.

This indicated that the model was starting to overfit.

### 8.3 EarlyStopping

To control overfitting, EarlyStopping was introduced.

```python
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)
```

**Explanation:**

- `monitor='val_loss'`: Tracks validation loss.
- `patience=3`: Stops training after three consecutive epochs without improvement.
- `restore_best_weights=True`: Restores the model weights from the epoch with the lowest validation loss.

The EarlyStopping experiment was trained for up to 20 epochs.

**Results:**

| Epoch | Training Accuracy | Validation Accuracy | Validation Loss |
|---:|---:|---:|---:|
| 1 | 94.17% | 91.23% | 0.2778 |
| 2 | 94.41% | 91.35% | 0.2679 |
| 3 | 94.72% | 91.75% | **0.2665** |
| 4 | 95.12% | 91.27% | 0.2801 |
| 5 | 95.25% | 91.15% | 0.2846 |
| 6 | 95.44% | 90.85% | 0.3023 |

Training stopped after Epoch 6 because validation loss had not improved for three consecutive epochs.

The best weights from Epoch 3 were restored.

**Final EarlyStopping validation performance:**

- Validation Accuracy: 91.75%
- Validation Loss: 0.2665

EarlyStopping helped select a checkpoint based on validation loss rather than simply using the final training epoch.

---

## 9. Model Evaluation

The CNN was evaluated using multiple classification metrics.

### 9.1 Confusion Matrix

A confusion matrix was generated to understand how accurately the model classified each clothing category.

The diagonal represents correct predictions, while values outside the diagonal represent misclassifications.

**Observations:**

- Trouser was classified very accurately.
- Bag, Sandal, Sneaker, and Ankle boot also achieved strong results.
- Shirt was the most challenging category.
- Shirt images were frequently confused with T-shirt/top, Pullover, and Coat.

### 9.2 Precision, Recall and F1-Score

**Precision:** Out of all images predicted as a particular category, how many were actually from that category?

**Recall:** Out of all actual images belonging to a category, how many did the model correctly identify?

**F1-Score:** Harmonic mean of precision and recall.

\[
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
\]

### Classification Report

The following results were obtained on the validation dataset.

| Category | Precision | Recall | F1-Score |
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

**Macro Average:**

- Precision: 0.92
- Recall: 0.92
- F1-Score: 0.92

**Weighted Average:**

- Precision: 0.92
- Recall: 0.92
- F1-Score: 0.92

### 9.3 Per-Class Accuracy

Based on the validation confusion matrix:

| Category | Correct Predictions | Accuracy |
|---|---:|---:|
| T-shirt/top | 536 / 600 | 89.33% |
| Trouser | 593 / 600 | 98.83% |
| Pullover | 527 / 600 | 87.83% |
| Dress | 536 / 600 | 89.33% |
| Coat | 512 / 600 | 85.33% |
| Sandal | 584 / 600 | 97.33% |
| Shirt | 469 / 600 | 78.17% |
| Sneaker | 583 / 600 | 97.17% |
| Bag | 591 / 600 | 98.50% |
| Ankle boot | 574 / 600 | 95.67% |

**Key Observation:**

Shirt had the lowest classification accuracy.

The model frequently confused Shirts with other visually similar clothing categories. This suggests that some clothing categories have overlapping visual characteristics, making them harder to distinguish.

---

## 10. CNN Feature Map Visualization

To understand what the CNN learns internally, feature maps were extracted from the convolution layers.

A feature map represents the activation produced by a particular filter when it processes an image.

### First Convolution Layer

The first convolution layer contains 32 filters.

Each filter produces a feature map of size 26 × 26.

The extracted output shape was:

```text
(1, 26, 26, 32)
```

These feature maps provide insight into the visual patterns detected by the first convolution layer.

### Second Convolution Layer

The second convolution layer contains 64 filters.

Its output shape was:

```text
(1, 11, 11, 64)
```

The second layer receives the output of the first pooling layer and learns more complex representations.

### Observations

- First-layer feature maps were generally easier to visually interpret.
- Second-layer feature maps appeared more abstract.
- Deeper layers combine earlier visual representations.
- Not every feature map corresponds to one recognizable object or clothing characteristic.

This experiment helped visualize how CNNs progressively transform raw images into learned feature representations.

---

## 11. Filter Weight Visualization

In addition to feature maps, the actual filter weights learned by the first convolution layer were extracted and visualized.

```python
filters, biases = cnn_model.layers[0].get_weights()
```

The extracted filter shape was:

```text
(3, 3, 1, 32)
```

This represents:

- Filter height: 3
- Filter width: 3
- Input channels: 1
- Number of filters: 32

Each filter contains nine learnable weights and one bias value.

The weights are initialized before training and updated through backpropagation and the optimizer.

Visualizing these filters provides insight into the learned parameters inside the CNN.

**Difference between Feature Maps and Filters:**

- Filters are the learned weights used to process images.
- Feature maps are the outputs generated when those filters are applied to images.

---

## 12. Experiment: CNN Without MaxPooling

To understand the importance of MaxPooling, another CNN architecture was created without pooling layers.

### Modified Architecture

```python
cnn_no_pool = Sequential([
    Input(shape=(28, 28, 1)),

    Conv2D(32, (3, 3), activation='relu'),

    Conv2D(64, (3, 3), activation='relu'),

    Flatten(),

    Dense(128, activation='relu'),

    Dropout(0.3),

    Dense(10, activation='softmax')
])
```

### Why was this experiment performed?

MaxPooling reduces spatial dimensions and computational requirements.

Removing it preserves more spatial information but increases the size of the feature representation.

### Architecture Comparison

| Metric | With MaxPooling | Without MaxPooling |
|---|---:|---:|
| First convolution output | 26 × 26 | 26 × 26 |
| After first pooling | 13 × 13 | 26 × 26 |
| Second convolution output | 11 × 11 | 24 × 24 |
| After second pooling | 5 × 5 | 24 × 24 |
| Flattened features | 1,600 | 36,864 |
| Dense layer parameters | 204,928 | 4,718,720 |

### Training Observations

The CNN without MaxPooling produced the following results:

| Epoch | Training Accuracy | Validation Accuracy | Validation Loss |
|---:|---:|---:|---:|
| 1 | 84.45% | 88.78% | 0.3097 |
| 2 | 89.85% | 90.15% | 0.2579 |
| 3 | 91.91% | 90.60% | **0.2517** |
| 4 | 93.68% | 90.83% | 0.2557 |
| 5 | 94.88% | **91.72%** | 0.2635 |
| 6 | 96.05% | 91.63% | 0.2689 |

**Observations:**

- Training accuracy continued increasing.
- Validation loss started increasing after Epoch 3.
- The best observed validation accuracy was 91.72% at Epoch 5.
- The best validation loss was 0.2517 at Epoch 3.
- The larger architecture required significantly more computation.

The experiment demonstrated that preserving more spatial information does not automatically improve classification performance.

MaxPooling helps reduce computational requirements and the number of parameters in subsequent layers.

The experiment also highlighted the importance of monitoring validation loss and controlling overfitting.

---

## 13. Final Model Comparison

The following table summarizes the performance of all the major models.

| Model | Validation Accuracy |
|---|---:|
| Logistic Regression | 85.53% |
| Decision Tree | 80.62% |
| Random Forest | 87.87% |
| CNN with MaxPooling | 91.75% |

### Final Test Performance

After selecting the CNN with MaxPooling and evaluating it on the separate test dataset:

| Metric | Result |
|---|---:|
| Test Accuracy | **91.93%** |
| Test Loss | **0.2479** |
| Total Test Images | 10,000 |
| Correctly Classified | Approximately 9,193 |
| Incorrectly Classified | Approximately 807 |

### Why CNN Was the Final Choice

CNN achieved the best validation accuracy among the tested models and performed well on the unseen test dataset.

Compared with Random Forest:

- Random Forest validation accuracy: 87.87%.
- CNN validation accuracy: 91.75%.
- Improvement: Approximately 3.88 percentage points.

CNN was particularly effective because it learned spatial and visual patterns directly from image data.

It also demonstrated better performance on difficult categories such as Shirt.

The final test accuracy of 91.93% indicates that the selected CNN generalized well to unseen Fashion MNIST images.

---

## 14. Key Learnings

This project provided practical experience with the complete machine learning and deep learning workflow.

### Machine Learning Concepts

- Supervised learning.
- Multiclass classification.
- Training, validation, and test datasets.
- Stratified sampling.
- Feature extraction and normalization.
- Model training and evaluation.
- Hyperparameter tuning.
- Overfitting and generalization.

### Traditional Machine Learning

- Logistic Regression.
- Decision Trees.
- Random Forest.
- Ensemble learning.
- Decision Tree depth optimization.
- Random Forest parameter tuning.

### Deep Learning

- Neural networks.
- Convolution operations.
- Filters and learnable weights.
- ReLU activation.
- MaxPooling.
- Flatten layers.
- Fully connected layers.
- Dropout regularization.
- Softmax classification.
- Backpropagation.
- Adam optimizer.
- Sparse Categorical Crossentropy.
- EarlyStopping.

### Model Evaluation

- Accuracy.
- Loss.
- Precision.
- Recall.
- F1-score.
- Confusion matrix.
- Per-class performance analysis.

### CNN Interpretability

- Feature map extraction.
- Convolution filter visualization.
- Understanding learned image representations.
- Comparing convolution layers.
- Understanding the impact of MaxPooling.

---

## 15. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| NumPy | Numerical operations and array manipulation |
| Pandas | Dataset loading and manipulation |
| Matplotlib | Data visualization and feature map plotting |
| Scikit-learn | Traditional ML models and evaluation |
| TensorFlow | Deep learning framework |
| Keras | CNN architecture and model training |
| Google Colab / Jupyter Notebook | Development environment |

---

## 16. How to Run the Project

### Prerequisites

Install the required libraries:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

### Steps

1. Clone or download the project repository.
2. Open the Jupyter Notebook or Google Colab notebook.
3. Make sure the Fashion MNIST CSV files are available.
4. Update the dataset file paths if required.
5. Run the notebook cells sequentially.
6. Execute preprocessing and exploratory data analysis.
7. Train and evaluate the traditional machine learning models.
8. Train the CNN and perform model evaluation.
9. Explore feature maps, filters, and the MaxPooling experiment.

### Important Preprocessing Note

The same normalization must be applied to training, validation, and test images.

For example:

```python
X_test_cnn = X_test_cnn / 255.0
```

The model was trained using pixel values between 0 and 1. Feeding unnormalized test images with values between 0 and 255 produced unreliable predictions and an unusually high loss.

Applying consistent preprocessing corrected the issue.

---

## 17. Conclusion

This project demonstrates the progression from traditional machine learning algorithms to deep learning for image classification.

Logistic Regression, Decision Tree, and Random Forest provided useful baselines and helped understand the limitations of traditional approaches when working with image data.

CNN proved to be more suitable for this task because it automatically learns spatial and visual features from images through convolution operations.

The project also explored model optimization, overfitting, EarlyStopping, feature visualization, filter weights, and the impact of MaxPooling.

The final CNN achieved **91.93% test accuracy** on 10,000 previously unseen Fashion MNIST images.

Beyond the final accuracy, the project provided practical understanding of how image classification models are developed, trained, evaluated, and improved.

**The primary takeaway:** Choosing the right model architecture for the data is just as important as tuning its parameters.

---

## Final Project Result

**Best-performing model:** Convolutional Neural Network (CNN)

**Dataset:** Fashion MNIST

**Number of classes:** 10

**Final test accuracy:** 91.93%

**Final test loss:** 0.2479

**Primary framework:** TensorFlow / Keras

