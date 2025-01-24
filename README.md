# Twitter Sentiment Analysis

This project focuses on analyzing the sentiment of tweets using machine learning. The goal is to classify tweets as either positive or negative using Natural Language Processing (NLP) techniques and machine learning models.

## Project Structure

├── training.1600000.processed.noemoticon.csv  # Dataset (link provided below)

├── app.py                                     # Streamlit web app

├── preparing_stemmed_data.ipynb               # Data preprocessing script

├── Train_model_2.ipynb                        # Model training script

├── README.md                                  # Project documentation

├── requirements.txt                           # Required dependencies


## Dataset

The dataset used for training the model contains 1.6 million tweets labeled as positive (1) or negative (0). You can download the dataset from the following link:

### Download Dataset - https://drive.google.com/file/d/1XoS6xebJJCk2ZD5qHZQxrC6-zlvQZBY9/view?usp=sharing

Installation

Follow these steps to set up and run the project:

### Clone the repository:

     git clone https://github.com/mohdajeem/Twitter-Sentiment-Analysis.git 
     
     cd twitter-sentiment-analysis

### Install dependencies:

pip install -r requirements.txt

# 
Download the Dataset from the link,
Run jupyter notebook and run jupyter files
1. preparing_stemmed_data.ipynb
2. Train_model_2.ipynb
3. Run the Streamlit app:
     streamlit run app.py

### Steps Involved

#### 1. Data Preprocessing (preparing_stemmed_data.ipynb)

    Load the dataset
    
    Clean the text (remove special characters, lowercase, remove stopwords)
    
    Apply stemming to reduce words to their root form
    
    Save the preprocessed data

#### 2. Model Training (Train_model_2.ipynb)

    Load preprocessed data
    
    Convert text into numerical format using TF-IDF Vectorizer
    
    Train a Logistic Regression model
    
    Save the trained model for later use

#### 3. Sentiment Prediction (app.py)

    Load the trained model
    
    Accept user input (tweet text)
    
    Preprocess and transform the input text
    
    Predict sentiment (positive or negative)
    
    Performance Metrics
    
    Model achieved an accuracy of 78% on the test data.
    
    Training and preprocessing took approximately 11 hours, including data cleaning and model training.

## Future Improvements

    Use deep learning models such as LSTMs for better performance.
    
    Optimize the data preprocessing pipeline to reduce processing time.
    
    Implement additional feature engineering techniques.

# Technologies Used

    Python
    
    Streamlit
    
    Scikit-learn
    
    NLTK
    
    Pandas
    
    NumPy

# Author:- 
    Mohd ajeem
