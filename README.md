# 🛰️ Geospatial Land Classification: CNN & ViT Hybrid Study
## Geospatial Land Classification Study


![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Expert-EE4C2C?logo=pytorch&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Hybrid%20Model-D00000?logo=keras&logoColor=white)
![IBM](https://img.shields.io/badge/IBM-Professional%20Capstone-052FAD?logo=ibm&logoColor=white)


### Project Overview
The capstone project, completed as part of the IBM Deep Learning with Keras and TensorFlow course, focuses on building and advanced land classification system for agricultural applications using satellite imagery.
The project simulates an AI Engineer role at a fertilizer company, where the core objective is to develop and rigorously compare state=of the=art deep learning models for accurately classifying terrain (e.g. crops, forests, water bodies)
The entire deep learning pipeline was implemented from custom geospatial data handling to comparative model analysis, showcasing expertise across leading deep learning frameworks.

### Key Technologies & Skills Demonstrated
| Category  | Skills & tools  |
|---|---|
| Deep Learning Frameworks  | Keras (TensorFlow) and PyTorch  |
| Model Architectures  | Convolutional Neural Networks (CNNs), Vision Transformers (ViT) and Hybrid CNN-ViT model  |
| Data Handling  | Geospatial Image Data Loading (memory-based vs. generator-based), Data Augmentation, Preprocessing  |
| Advanced Techniques  | Transfer Learning (fine-tuning pre-trained models) |
| Performance Evaluation  | Accuracy, Precision, Recall, F1​-score, AU-ROC, Confusion Matrix  |
| Deliverables  | Jupyter Notebooks (technical rigor)  |

### Deliverables and Core Achievements
1. Robust Deep Learning Model Development

    CNN Implementation: Developed and trained independent CNN models using both Keras and PyTorch to solve the land classification problem.

    Vision Transformer Integration:  Designed and implemented a hybrid deep learning model by integrating features from pre-trained CNNs and Vision Transformers. The entire combined architecture was then fine-tuned to optimize performance for the agricultural land classification task.

    Comparative Analysis: Conducted a comprehensive comparative study of CNNs and Hybrid CNN-Vision Transformer performance across the two major frameworks.

2. Full-Cycle Deep Learning Pipeline

    Data Handling: Implemented efficient techniques for geospatial image dataset loading and applied custom data augmentation strategies in both Keras and PyTorch.

    Model Evaluation: Rigorously evaluated all models using a suite of quantitative metrics, including F1​-score and AU-ROC, to ensure robust and reliable performance for a real-world application.

### 📂 Repository Contents

The study is organized into a sequential 9-step pipeline:

- **01-03: Data Engineering** – Geospatial loading, custom augmentation, and preprocessing.
- **04-06: CNN Development** – Implementing and evaluating baselines in Keras and PyTorch.
- **07-09: Advanced Integration** – Master-level implementation of Vision Transformers (ViT) and a custom Hybrid CNN-ViT integration.

## 📊 Dataset
The project utilizes the **EuroSAT-style Geospatial Dataset** (Land Use and Land Cover Classification). 
The raw data is fetched automatically within the notebooks from the public IBM Cloud Object Storage:
- **Source URL:** [images-dataSAT.tar](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5vTzHBmQUaRNJQe5szCyKw/images-dataSAT.tar)


📜 Attributions & License
This project was developed as a Capstone for the IBM Deep Learning Professional Certificate. The core datasets and initial lab structures are provided by IBM Skills Network under their educational terms. All model implementations, hybrid architecture integration, and comparative analyses were performed by me as part of this study.