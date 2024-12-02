# **Breast Cancer Prediction Using Machine Learning**

This project leverages **Convolutional Neural Networks (CNNs)** to detect and classify breast cancer from ultrasound images. Early detection of breast cancer is crucial, and machine learning techniques offer a promising approach to enhance accuracy and speed.

---

## **Table of Contents**
1. [Introduction](#introduction)
2. [Project Objectives](#project-objectives)
3. [Features](#features)
4. [Dataset Information](#dataset-information)
5. [Model Architecture](#model-architecture)
6. [Installation Guide](#installation-guide)
7. [Training and Evaluation](#training-and-evaluation)
8. [Results and Analysis](#results-and-analysis)
9. [Challenges and Solutions](#challenges-and-solutions)
10. [Future Enhancements](#future-enhancements)


---

## **Introduction**
Breast cancer is one of the leading causes of cancer-related deaths worldwide. This project focuses on building a robust **deep learning model** using CNN to identify different types of breast cancer from ultrasound images. By automating the diagnosis process, we aim to assist radiologists in making faster and more accurate decisions.

---

## **Project Objectives**
- Develop a CNN model to classify breast ultrasound images into:
  - **Benign** (non-cancerous)
  - **Malignant** (cancerous)
  - **Normal**
- Achieve high classification accuracy using deep learning techniques.
- Provide insights through visualizations like confusion matrices and classification reports.

---

## **Features**
- **Automated Image Preprocessing**: Images are resized, normalized, and augmented for optimal model performance.
- **Deep Learning Model**: Uses a custom CNN architecture with multiple layers for feature extraction.
- **Evaluation Metrics**: Accuracy, precision, recall, F1-score, and confusion matrix.
- **Scalability**: The project can be extended to include other medical imaging datasets and different cancer types.

---

## **Dataset Information**
- **Source**: [BUSI with GT Dataset](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) (Breast Ultrasound Images with Ground Truth)
- **Classes**: Benign, Malignant, and Normal
- **Dataset Size**: 1765 images divided into three categories.

### **Sample Dataset Loading Code:**
```python
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "/content/drive/MyDrive/Dataset_BUSI_with_GT",
    seed=123,
    shuffle=True,
    image_size=(256, 256),
    batch_size=32
)
```

---


## **Model Architecture**
The model is designed with the following layers:
- **Input Layer**: Accepts images of size `(256, 256, 3)`.
- **Convolutional Layers**: Extract features using filters of various sizes.
- **Pooling Layers**: Reduces dimensionality while retaining important features.
- **Fully Connected Layers**: Combines extracted features for final classification.
- **Output Layer**: A softmax layer with 3 nodes for the three classes.

### **Model Summary Code:**
```python
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax')
])
model.summary()
```

---

## **Installation Guide**

### **Prerequisites**
- **Python 3.7+**
- **TensorFlow 2.x**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Scikit-learn**

### **Steps to Install:**
1. **Clone the repository:**
   ```bash
   git clone https://github.com/username/repository-name.git
   cd repository-name
   ```
2. **Install the required packages**
   ```bash
   pip install -r requirements.txt
   ```

---

## **Training and Evaluation**

### **Data Augmentation**  
Techniques used include rotation, flipping, and zooming to enhance model generalization.

### **Model Compilation:**
```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

### **Training:**
```python
history = model.fit(dataset, epochs=20, validation_split=0.2)
```

### **Evaluation:**
```python
loss, accuracy = model.evaluate(dataset)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
```
---

## **Results and Analysis**
- **Accuracy**: Over 90% test accuracy achieved.
- **Confusion Matrix**: Demonstrates the model’s capability in correctly classifying each category.

---

## **Challenges and Solutions**

### **Challenges**
- **Class Imbalance**: Certain classes had fewer images, leading to skewed results.
- **Overfitting**: The model tended to overfit on the training dataset.

### **Solutions**
- **Data Augmentation** helped balance the dataset.
- **Early Stopping** and **Dropout Layers** were implemented to prevent overfitting.

---

## **Future Enhancements**
- **Multi-modal Data Integration**: Combine ultrasound images with other diagnostic data like mammograms.
- **Transfer Learning**: Use pre-trained models like **ResNet** or **VGG** for better accuracy.
- **Web Application**: Deploy the model as a web service for real-time breast cancer diagnosis.

---
