# Iris Dataset Classification with Multi-Layer Perceptron (MLP)

A deep learning project implementing a Multi-Layer Perceptron (MLP) architecture using TensorFlow/Keras to classify Iris flower species based on physical dimensions.

## 📌 Project Overview

This repository demonstrates the end-to-end pipeline of building, training, and evaluating a neural network on the classic **Iris dataset**. Features are standardized using `StandardScaler`, and a deep `Sequential` model is constructed to perform multi-class classification into three target species: *Setosa*, *Versicolor*, and *Virginica*.

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.x
* **Deep Learning Framework:** TensorFlow / Keras
* **Machine Learning & Preprocessing:** Scikit-Learn
* **Data Manipulation:** NumPy, Pandas
* **Visualization:** Matplotlib

### Required Packages

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

## 🧠 Model Architecture

The model is built using TensorFlow's `Sequential` API with the following structure:

| Layer | Type | Neurons / Units | Activation |
| ----- | ----- | ----- | ----- |
| Input | Dense | 64 | ReLU |
| Hidden 1 | Dense | 128 | ReLU |
| Hidden 2 | Dense | 256 | ReLU |
| Hidden 3 | Dense | 512 | ReLU |
| Output | Dense | 3 | Softmax |

* **Optimizer:** Adam (Learning Rate: `0.0001`)
* **Loss Function:** `sparse_categorical_crossentropy`
* **Batch Size:** 1
* **Epochs:** 15

## 🚀 How to Run

1. Clone this repository:
```bash
git clone https://github.com/sedaygit/Iris_Dataset_MLP.git
cd Iris_Dataset_MLP
```
2. Run the script:
```bash
python iris_dataset_mlp.py
```
## 📊 Evaluation & Metrics

The script performs full evaluation on the hold-out test dataset (30% split) and outputs:

* **Accuracy Score & Loss**
* **Detailed Classification Report** (Precision, Recall, F1-Score)
* **Confusion Matrix**
* **Training vs. Validation Plots** for Accuracy and Loss curves over epochs

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
