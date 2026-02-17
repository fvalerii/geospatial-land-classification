# 🛰️ Geospatial Land Classification: CNN & ViT Hybrid Study
### *Geospatial Land Classification Study*


![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Expert-EE4C2C?logo=pytorch&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Hybrid%20Model-D00000?logo=keras&logoColor=white)
![IBM](https://img.shields.io/badge/IBM-Professional%20Capstone-052FAD?logo=ibm&logoColor=white)

---

## 📋 Project Overview
The capstone project, completed as part of the IBM Deep Learning Professional Certificate, focuses on building an advanced land classification system for agricultural applications using satellite imagery. 

The project simulates an AI Engineer role at a fertilizer company, where the core objective is to develop and rigorously compare state-of-the-art deep learning models for accurately classifying terrain (e.g., crops, forests, water bodies). The entire deep learning pipeline was implemented—from custom geospatial data handling to comparative model analysis—showcasing expertise across leading deep learning frameworks.

---

## Final Results

![Hybrid Model Comparison](images/comparison_table.png)

1. **Robust Deep Learning Model Development**

    **CNN Implementation:** Developed and trained independent CNN models using both Keras and PyTorch to solve the land classification problem.

    **Vision Transformer Integration:**  Designed and implemented a hybrid deep learning model by integrating features from pre-trained CNNs and Vision Transformers. The entire combined architecture was then fine-tuned to optimize performance for the agricultural land classification task.

    **Comparative Analysis:** Conducted a comprehensive comparative study of CNNs and Hybrid CNN-Vision Transformer performance across the two major frameworks.

2. **Full-Cycle Deep Learning Pipeline**

    **Data Handling:** Implemented efficient techniques for geospatial image dataset loading and applied custom data augmentation strategies in both Keras and PyTorch.

    **Model Evaluation:** Rigorously evaluated all models using a suite of quantitative metrics, including F1​-score and AU-ROC, to ensure robust and reliable performance for a real-world application.


---

## 📂 Repository Contents

### 🚀 Research Pipeline & Notebooks

| # | Phase | Technical Focus | Colab Access |
|:--|:------|:----------------|:------------:|
| 01 | **Data Engineering** | Memory-Based vs. Generator-Based Ingestion | [Launch 🚀](https://colab.research.google.com/github/fvalerii/geospatial-land-classification/blob/main/notebooks/01_data_handling/01_memory_vs_generator_loading.ipynb) |
| 02 | **Data Engineering** | Scalable Augmentation Strategies (TensorFlow) | [Launch 🚀](https://colab.research.google.com/github/fvalerii/geospatial-land-classification/blob/main/notebooks/01_data_handling/02_keras_data_augmentation.ipynb) |
| 03 | **Data Engineering** | Torchvision Pipeline & Tensor Transformations | [Launch 🚀](https://colab.research.google.com/github/fvalerii/geospatial-land-classification/blob/main/notebooks/01_data_handling/03_pytorch_data_augmentation.ipynb) |
| 04 | **CNN Development** | Keras Convolutional Baseline & Optimization | [Launch 🚀](https://colab.research.google.com/github/fvalerii/geospatial-land-classification/blob/main/notebooks/02_cnn_development/04_keras_cnn_classifier.ipynb) |
| 05 | **CNN Development** | PyTorch Implementation & State Dict Management | [Launch 🚀](https://colab.research.google.com/github/fvalerii/geospatial-land-classification/blob/main/notebooks/02_cnn_development/05_pytorch_cnn_classifier.ipynb) |
| 06 | **Analysis** | Cross-Framework Performance Benchmarking | [Launch 🚀](https://colab.research.google.com/github/fvalerii/geospatial-land-classification/blob/main/notebooks/02_cnn_development/06_cnn_comparative_analysis.ipynb) |
| 07 | **Hybrid Integration** | Vision Transformers (ViT) in Keras | [Launch 🚀](https://colab.research.google.com/github/fvalerii/geospatial-land-classification/blob/main/notebooks/03_hybrid_transformer_integration/07_keras_vision_transformer.ipynb) |
| 08 | **Hybrid Integration** | Vision Transformers (ViT) in PyTorch | [Launch 🚀](https://colab.research.google.com/github/fvalerii/geospatial-land-classification/blob/main/notebooks/03_hybrid_transformer_integration/08_pytorch_vision_transformer.ipynb) |
| 09 | **Final Study** | **Hybrid CNN-ViT Model Integration** | **[Launch 🏆](https://colab.research.google.com/github/fvalerii/geospatial-land-classification/blob/main/notebooks/03_hybrid_transformer_integration/09_hybrid_cnn_vit_integration.ipynb)** |

---

## 📊 Dataset
The project utilizes the **EuroSAT-style Geospatial Dataset** (Land Use and Land Cover Classification). 
The raw data is fetched automatically within the notebooks from the public IBM Cloud Object Storage:
- **Source URL:** [images-dataSAT.tar](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5vTzHBmQUaRNJQe5szCyKw/images-dataSAT.tar)

---

## ⚙️ Execution Guide

### **Option A: Colab Execution (Cloud)**
Click on the link in the Colab Access tab in the table [🚀 Research Pipeline & Notebooks](### 🚀 Research Pipeline & Notebooks).

### **Option B: Local Execution (WSL2/GPU)**
Recommended for leveraging local GPU acceleration.

### **1. Environment Setup** 
It is recommended to use an environment with Python 3.12.8:
##### Using Conda (Recommended):
```bash
conda env create -f environment.yml
conda activate vit-research
```
##### Using Pip:
```bash
pip install -r requirements.txt
```
### **2. Run the Research Study**
Navigate to the `notebooks/` directory and launch the modules via VS Code or Jupyter Lab.

---

## 💻 Tech Stack
- **Deep Learning Frameworks:** Keras (TensorFlow) and PyTorch.
- **Model Architectures:** Convolutional Neural Networks (CNNs), Vision Transformers (ViT) and Hybrid CNN-ViT model.
- **Data Handling:** Geospatial Image Data Loading (memory-based vs. generator-based), Data Augmentation, Preprocessing.
- **Advanced Techniques:** Transfer Learning (fine-tuning pre-trained models).
- **Performance Evaluation:** Accuracy, Precision, Recall, F1​-score, AU-ROC, Confusion Matrix.
- **| Deliverables:** Jupyter Notebooks (technical rigor)

---

📜 Attributions & License
This project was developed as a Capstone for the IBM Deep Learning Professional Certificate. The core datasets and initial lab structures are provided by IBM Skills Network under their educational terms. All model implementations, hybrid architecture integration, and comparative analyses were performed by me as part of this study.