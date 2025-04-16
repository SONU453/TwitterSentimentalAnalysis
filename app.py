import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
# nltk.data.path.append('C:/Users/aditi gupta/nltk_data')
# nltk.download('stopwords', download_dir='C:/Users/aditi gupta/nltk_data')
# nltk.download('punkt', download_dir='C:/Users/aditi gupta/nltk_data')
# nltk.download('wordnet', download_dir='C:/Users/aditi gupta/nltk_data')

# import nltk
nltk.data.path.append('C:/Users/aditi gupta/nltk_data')

# Load the model and vectorizer
model_filename = 'pred_trained_model.sav'
vectorizer_filename = 'tfidf_vectorizer.pkl'  # Save the vectorizer as well

# Load model
model = pickle.load(open(model_filename, 'rb'))

# Load vectorizer
vectorizer = pickle.load(open(vectorizer_filename, 'rb'))

# NLTK resources for preprocessing
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Custom stopwords list
words_to_keep = {"no", "not", "never", "nor", "none", "nobody", "nothing", 
                 "neither", "nowhere", "isn't", "wasn't", "aren't", 
                 "weren't", "don't", "doesn't", "didn't", "won't", 
                 "wouldn't", "shouldn't", "can't", "cannot", "couldn't", 
                 "very", "really", "extremely", "absolutely", "totally", 
                 "completely", "highly", "so", "quite", "too", "enough", 
                 "particularly", "especially", "most", "just", "only", 
                 "barely", "hardly", "scarcely", "somewhat", "slightly", 
                 "less", "might", "may", "could", "would", "should", 
                 "but", "however", "although", "though", "yet", "still", 
                 "nevertheless", "even", "like", "love", "hate", "enjoy", 
                 "prefer", "dislike", "amazing", "awesome", "bad", 
                 "terrible", "horrible", "wonderful", "fantastic", "awful", 
                 "great", "good", "want", "need", "wish", "hope", "prefer", 
                 "expect", "demand", "appreciate", "despise", "hardly", 
                 "scarcely", "rarely", "little", "maybe", "perhaps", 
                 "probably", "possibly", "definitely", "certainly", "surely", 
                 "undoubtedly"}

stop_words = stop_words.difference(words_to_keep)

def preprocess_text(content):
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower()
    content = word_tokenize(content)
    content = [lemmatizer.lemmatize(word) for word in content if word not in stop_words]
    return ' '.join(content)

# Streamlit UI
st.title("Sentiment Analysis on Tweets")
st.write("Enter a tweet to analyze its sentiment.")

# Text input for tweet
user_input = st.text_area("Tweet", "")

# Button to predict
if st.button("Predict Sentiment"):
    if user_input:
        processed_input = preprocess_text(user_input)
        input_vector = vectorizer.transform([processed_input])
        
        # Make prediction
        prediction = model.predict(input_vector)
        
        # Display result
        if prediction[0] == 1:
            st.success("Predicted Sentiment: Positive")
        else:
            st.error("Predicted Sentiment: Negative")
    else:
        st.warning("Please enter a tweet.")

# Run the app using the command: streamlit run app.py 
