import streamlit as st
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Set page configuration
st.set_page_config(
    page_title="Vaccination Tweet Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')

download_nltk_data()
stop_words = set(stopwords.words('english'))

# Preprocessing functions
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"https\S+|www\S+https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^\w\s]','', text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)

def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity < 0:
        return "Negative"
    elif polarity == 0:
        return "Neutral"
    else:
        return "Positive"

# Main function
def main():
    st.title("🔍 Vaccination Tweets Sentiment Analysis")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", ["Upload & Process", "Analysis", "Prediction"])

    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'vectorizer' not in st.session_state:
        st.session_state.vectorizer = None

    if page == "Upload & Process":
        st.header("📤 Upload Your Data")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file:
            try:
                with st.spinner('Reading and processing data...'):
                    df = pd.read_csv(uploaded_file)
                    
                    if 'text' not in df.columns:
                        st.error("❌ CSV file must contain a 'text' column!")
                        return
                    
                    # Process data
                    df['processed_text'] = df['text'].apply(preprocess_text)
                    df['sentiment'] = df['processed_text'].apply(get_sentiment)
                    
                    st.session_state.data = df
                    st.success("✅ Data processed successfully!")
                    
                    # Display sample
                    st.subheader("Sample of processed data:")
                    st.dataframe(df.head())
                    
                    # Display basic statistics
                    st.subheader("Basic Statistics:")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Tweets", len(df))
                    with col2:
                        st.metric("Positive Tweets", len(df[df['sentiment']=='Positive']))
                    with col3:
                        st.metric("Negative Tweets", len(df[df['sentiment']=='Negative']))
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    elif page == "Analysis":
        st.header("📊 Data Analysis")
        
        if st.session_state.data is None:
            st.warning("⚠️ Please upload and process data first!")
            return
        
        # Sentiment Distribution
        st.subheader("Sentiment Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=st.session_state.data, x='sentiment', palette='viridis')
        plt.title("Distribution of Sentiments")
        st.pyplot(fig)
        
        # Word Clouds
        st.subheader("Word Clouds")
        sentiment_choice = st.selectbox(
            "Choose sentiment to visualize",
            ["Positive", "Negative", "Neutral"]
        )
        
        filtered_text = ' '.join(
            st.session_state.data[
                st.session_state.data['sentiment'] == sentiment_choice
            ]['processed_text']
        )
        
        if filtered_text.strip():
            wordcloud = WordCloud(
                width=800, 
                height=400,
                background_color='white',
                colormap='viridis'
            ).generate(filtered_text)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            plt.imshow(wordcloud)
            plt.axis('off')
            st.pyplot(fig)
        
        # Train Model
        if st.button("🎯 Train Model"):
            with st.spinner("Training model..."):
                vectorizer = CountVectorizer()
                X = vectorizer.fit_transform(st.session_state.data['processed_text'])
                y = st.session_state.data['sentiment']
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train, y_train)
                
                st.session_state.model = model
                st.session_state.vectorizer = vectorizer
                
                accuracy = model.score(X_test, y_test)
                st.success(f"✅ Model trained! Accuracy: {accuracy:.2f}")

    elif page == "Prediction":
        st.header("🎯 Sentiment Prediction")
        
        if st.session_state.model is None:
            st.warning("⚠️ Please train the model first!")
            return
        
        user_input = st.text_area(
            "Enter text for sentiment analysis:",
            height=100
        )
        
        if st.button("Predict Sentiment"):
            if user_input:
                with st.spinner("Analyzing..."):
                    # Process and predict
                    processed_input = preprocess_text(user_input)
                    vectorized_input = st.session_state.vectorizer.transform([processed_input])
                    prediction = st.session_state.model.predict(vectorized_input)[0]
                    probabilities = st.session_state.model.predict_proba(vectorized_input)[0]
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"Predicted Sentiment: {prediction}")
                    
                    with col2:
                        # Create probability dataframe
                        prob_df = pd.DataFrame({
                            'Sentiment': st.session_state.model.classes_,
                            'Probability': probabilities
                        })
                        st.write("Confidence Scores:")
                        st.dataframe(prob_df)
            else:
                st.warning("⚠️ Please enter some text!")

if __name__ == "__main__":
    main()