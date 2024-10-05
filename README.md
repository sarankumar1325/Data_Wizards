# Autism Detection using Machine Learning
This project builds a machine learning model to detect autism based on text data, using a dataset titled **autism_data.csv**. The model is trained and optimized using **Intel OneAPI extension for scikit-learn**, which increases processing speed by **1.73x**.

## Dataset
The dataset used for training the model is **autism_data.csv**, which contains features such as:
- Age
- Gender
- Family History
- Social Skills
- Communication Behavior
- Others related to autism detection

## Libraries and Tools
- Python
- Scikit-learn (with Intel OneAPI extension for speed optimization)
- Intel OneAPI

## Key Steps
1. **Data Loading**: Load the dataset `autism_data.csv`.
2. **Data Preprocessing**: Handle missing values, normalize the data, and split it into training and testing sets.
3. **Model Building**: Train a logistic regression model to predict autism.
4. **Speed Optimization**: Apply Intel OneAPI extension to accelerate the training time by **1.73x**.
5. **Evaluation**: Measure model performance (accuracy, precision, recall) and compare with baseline models.

## Results
The model achieves a performance improvement in both accuracy and speed, achieving **87% accuracy** and a **1.73x faster training time** with Intel OneAPI.

![img-1 ](https://github.com/user-attachments/assets/ca2f8265-8f21-48e1-92c1-d1c994dab43e)


## Files
- `Autism_Detection_with_ML.ipynb`: The notebook for building and training the machine learning model.
- `autism_data.csv`: The dataset used for training the model.

## Conclusion
This notebook demonstrates how Intel OneAPI can be effectively used to accelerate machine learning models and improve the detection of autism.

# Deep Learning for Autism Detection using CNN
This project develops a deep learning model using Convolutional Neural Networks (CNN) to classify images for autism detection. The model is enhanced with **Intel OneAPI for TensorFlow** for optimized training performance.

## Model Overview
The deep learning model is built using a **CNN architecture** to classify whether an image indicates autism or not. The model is then saved as weights to be used in a separate application file.

## Dataset
- Image dataset containing features related to autism, preprocessed for input to the CNN.

## Libraries and Tools
- Python
- TensorFlow (with Intel OneAPI extension)
- Intel OneAPI

## Key Steps
1. **Data Preprocessing**: Load the image dataset, resize, normalize, and split into training and validation sets.
2. **Model Building**: A CNN is built to learn image features related to autism detection.
3. **Speed Optimization**: Leverage Intel OneAPI for TensorFlow to optimize training time and model performance.
4. **Training**: The model is trained using the preprocessed data and optimized for classification accuracy.
5. **Saving Weights**: After training, the model’s weights are saved for use in the application (`app.py`).

## Results
The CNN model shows excellent performance in detecting autism with image data. The Intel OneAPI extension increases training speed significantly.

![ig -2 ](https://github.com/user-attachments/assets/85bffa91-f743-4506-939d-24667176cef6)



## Files
- `deep_learning.ipynb`: The deep learning notebook implementing the CNN.
- `app.py`: Application that uses the trained CNN model’s weights to classify images.
- Model Weights: Saved weights from the trained CNN model for further use.

## Conclusion
The CNN model, optimized with Intel OneAPI, performs well in detecting autism from images, enabling its integration into real-time applications for autism detection.
