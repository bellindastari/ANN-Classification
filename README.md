# **Breast Cancer Classification Using Neural Networks**  

This repository contains the implementation of a **Neural Network-based classification model** for identifying breast cancer types (benign or malignant). The project is developed as part of a **Data Science course final assignment**, utilizing **machine learning techniques** to solve a real-world classification problem.  

## **Project Overview**  
Breast cancer is one of the most prevalent cancers affecting women worldwide. Early detection and accurate classification of tumors are crucial for effective treatment. This project applies **Artificial Neural Networks (ANN)** to classify breast cancer using a dataset that includes **32 predictive features**, such as **radius, texture, perimeter, area, and smoothness**.  

The model is built using **Python** and implemented on **Google Colaboratory** with libraries such as **TensorFlow and Keras**. The final model achieves an accuracy of **93.85%**, demonstrating the effectiveness of neural networks in medical diagnosis.  

## **Dataset**  
The dataset used in this project is the **Breast Cancer Wisconsin Dataset**, available from the **UCI Machine Learning Repository**. It consists of **569 samples and 32 features**, representing various characteristics of cell nuclei present in a digitized breast mass image.  

## **Implementation**  
The project includes the following steps:  
1. **Data Preprocessing**: Cleaning, normalization, and feature selection.  
2. **Model Development**: Implementing a **Neural Network** using **TensorFlow and Keras**.  
3. **Training and Evaluation**: Splitting data into training and test sets, training the model, and evaluating its performance.  
4. **Hyperparameter Tuning** (future improvement): Optimizing the model for better accuracy.  

## **Requirements**  
To run the project, install the following dependencies:  
- Python 3.x  
- TensorFlow  
- Keras  
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-learn  

Install all dependencies using:  
```bash
pip install tensorflow keras numpy pandas matplotlib scikit-learn
```  

## **How to Use**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/breast-cancer-classification.git
   cd breast-cancer-classification
   ```  
2. Open the Jupyter Notebook or Google Colab.  
3. Run the **main script** to train and evaluate the model.  

## **Results**  
The trained **Neural Network model** achieves:  
✅ **93.85% accuracy** in classifying malignant and benign tumors.  
✅ Uses **28 most relevant features** for improved classification.  
✅ Can be further improved with **hyperparameter tuning**.  

## **Paper Reference**  
For a detailed explanation of the methodology and results, please refer to the full research paper:  
📄 **[Klasifikasi Jenis Kanker Payudara Menggunakan Pendekatan Model pada Machine Learning](./Klasifikasi_Jenis_Kanker_Payudara.pdf)**  

## **Contributors**  
- **Deanarani Kharisma**  
- **Dimas Satrio Adjie**  
- **Roswita Bellinda**  
- **Salsabila H. Sastro**  
