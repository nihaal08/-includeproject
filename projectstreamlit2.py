import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import emoji
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import nltk
from concurrent.futures import ThreadPoolExecutor
import time
from wordcloud import WordCloud
from googletrans import Translator

st.set_page_config(layout="wide")
nltk.download('punkt_tab',quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)

try:
    STOPWORDS = set(stopwords.words('english'))
except Exception as e:
    st.error(f"Error loading stopwords: {e}")
    STOPWORDS = set()

def load_css():
    st.markdown(
        """
        <style>
            body {
                color: red;               
                font-family: Arial, sans-serif;
            }
            h1, h2, h3, h4, h5, h6 {
                color: white;               
                text-transform: uppercase; 
            }
            .stSidebar {
                background-color: black;  
                color: darkred;                
            }
            .stButton > button {
                background-color: white;    
                color: black;             
                border: none;             
                padding: 10px;           
                border-radius: 5px;      
                cursor: pointer;         
                width: 100%;             
                text-align: center;      
                font-size: 16px;         
            }
            .stButton > button:hover {
                background-color: red;
                color: black;
            }
            .stTextInput > div > input {
                background-color: black;   
                color: red;                
                border: 1px solid red;    
            }
            .stTextArea > div > textarea {
                background-color: black;  
                color: red;               
                border: 1px solid red;    
            }
            table {
                background-color: black;
                color: red;
            }
            th {
                background-color: darkred;
                color: white;
            }
            td {
                background-color: black;
                color: red;
            }
            .stSpinner {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100%;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

load_css()

def initialize_scraped_database():
    conn = sqlite3.connect('scraped_sentiment_analysis.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scraped_reviews (
            id INTEGER PRIMARY KEY,
            name TEXT,
            rating TEXT,
            title TEXT,
            description TEXT,
            sentiment TEXT, 
            translated_description TEXT  
        )
    ''')
    conn.commit()
    conn.close()

def initialize_uploaded_database():
    conn = sqlite3.connect('uploaded_sentiment_analysis.db')
    cursor = conn.cursor()
    
    # Drop the existing table if necessary
    cursor.execute('DROP TABLE IF EXISTS uploaded_reviews')

    # Create a new table with the correct structure
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS uploaded_reviews (
            id INTEGER PRIMARY KEY,
            product_id TEXT,
            user_id TEXT,
            profile_name TEXT,
            helpfulness_numerator INTEGER,
            helpfulness_denominator INTEGER,
            score INTEGER,
            time INTEGER,
            summary TEXT,
            text TEXT,
            sentiment TEXT  
        )
    ''')
    conn.commit()
    conn.close()

def translate_text(text, target_language='en'):
    translator = Translator()
    try:
        translated = translator.translate(text, dest=target_language)
        return translated.text
    except Exception as e:
        st.error(f"Translation Error: {e}")
        return text

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def filter_unwanted_comments(reviews, unwanted_keywords):
    filtered_reviews = []
    for review in reviews:
        if not any(keyword.lower() in review['Description'].lower() for keyword in unwanted_keywords):
            filtered_reviews.append(review)
    return filtered_reviews

def get_request_headers():
    return {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'
    }

def single_page_scrape(url, page_number):
    reviews = []
    try:
        response = requests.get(f"{url}&pageNumber={page_number}", headers=get_request_headers())
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        boxes = soup.select('div[data-hook="review"]')
        
        for box in boxes:
            review = {
                'Name': box.select_one('[class="a-profile-name"]').text if box.select_one('[class="a-profile-name"]') else 'N/A',
                'Rating': box.select_one('[data-hook="review-star-rating"]').text.split(' out')[0] if box.select_one('[data-hook="review-star-rating"]') else 'N/A',
                'Title': box.select_one('[data-hook="review-title"]').text.strip() if box.select_one('[data-hook="review-title"]') else 'N/A',
                'Description': box.select_one('[data-hook="review-body"]').text.strip() if box.select_one('[data-hook="review-body"]') else 'N/A',
            }
            review['Title'] = re.sub(r'\d\.\d out of \d stars?', '', review['Title']).strip()
            review['Description'] = re.sub(r'Read more', '', review['Description'], flags=re.IGNORECASE).strip()
            review['Description'] = clean_text(review['Description'])
            
            # Translate description and log the output
            review['Translated_Description'] = translate_text(review['Description'], target_language='en')
            # Analyze sentiment on the translated text directly
            review['Sentiment'] = analyze_sentiment(review['Translated_Description'])
            
            reviews.append(review)
    except requests.exceptions.RequestException as e:
        st.error(f"Error on page {page_number}: {e}")
    return reviews

@st.cache_data(show_spinner=False)
def scrape_reviews(url, pages):
    reviews = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for page_number in range(1, pages + 1):
            futures.append(executor.submit(single_page_scrape, url, page_number))
            time.sleep(1)

        for future in futures:
            reviews.extend(future.result())
    
    unwanted_keywords = ['fake', 'unverified', 'not helpful', 'spam']
    reviews = filter_unwanted_comments(reviews, unwanted_keywords)
    return reviews

lem = WordNetLemmatizer()

@st.cache_data(show_spinner=False)
def preprocess_text(text):
    text = emoji.demojize(text)
    text = clean_text(text)
    tokens = word_tokenize(text.lower())
    cleaned_tokens = [lem.lemmatize(token) for token in tokens if token not in STOPWORDS]
    return ' '.join(cleaned_tokens)

analyzer = SentimentIntensityAnalyzer()

@st.cache_data(show_spinner=False)
def analyze_sentiment(text):
    sentiment_scores = analyzer.polarity_scores(text)
    compound = sentiment_scores['compound']

    if compound > 0.05:
        return 'Positive'
    elif compound < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def generate_insights(data):
    positive_reviews = data[data['Sentiment'] == 'Positive']
    negative_reviews = data[data['Sentiment'] == 'Negative']
    neutral_reviews = data[data['Sentiment'] == 'Neutral']

    insights = []
    insights.append(f"{len(positive_reviews)} positive reviews found.")
    insights.append(f"{len(negative_reviews)} negative reviews found.")
    insights.append(f"{len(neutral_reviews)} neutral reviews found.")
    return insights

def insert_scraped_review(name, rating, title, description, sentiment, translated_description):
    try:
        conn = sqlite3.connect('scraped_sentiment_analysis.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO scraped_reviews (name, rating, title, description, sentiment, translated_description) 
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (name, rating, title, description, sentiment, translated_description))
        conn.commit()
    except Exception as e:
        st.error(f"Error inserting review into scraped_reviews: {e}")
    finally:
        conn.close()

def insert_uploaded_review(product_id, user_id, profile_name, helpfulness_numerator, helpfulness_denominator,
                            score, time, summary, text, sentiment):
    try:
        conn = sqlite3.connect('uploaded_sentiment_analysis.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO uploaded_reviews (product_id, user_id, profile_name, helpfulness_numerator, 
                                          helpfulness_denominator, score, time, summary, text, sentiment) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (product_id, user_id, profile_name, helpfulness_numerator, helpfulness_denominator,
              score, time, summary, text, sentiment))
        conn.commit()
    except Exception as e:
        st.error(f"Error inserting review into uploaded_reviews: {e}")
    finally:
        conn.close()

def fetch_all_reviews(table_name):
    db_name = 'scraped_sentiment_analysis.db' if 'scraped' in table_name else 'uploaded_sentiment_analysis.db'
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute(f'SELECT * FROM {table_name}')
        data = cursor.fetchall()
    except Exception as e:
        st.error(f"Error fetching reviews from {table_name}: {e}")
        return []
    finally:
        conn.close()
    return data

def clear_database():
    try:
        conn = sqlite3.connect('scraped_sentiment_analysis.db')
        cursor = conn.cursor()
        cursor.execute('DELETE FROM scraped_reviews')
        conn.commit()

        conn = sqlite3.connect('uploaded_sentiment_analysis.db')
        cursor = conn.cursor()
        cursor.execute('DELETE FROM uploaded_reviews')
        conn.commit()

        st.success("All reviews have been cleared.")
    except Exception as e:
        st.error(f"Error clearing database: {e}")
    finally:
        conn.close()

initialize_scraped_database()
initialize_uploaded_database()

def display_navbar():
    st.sidebar.image('logo3.jpg', use_column_width=True)
    if st.sidebar.button("Home", key="home_button"):
        st.session_state.page = "Home"
    if st.sidebar.button("User Operations", key="user_operations_button"):
        st.session_state.page = "User Operations"
    if st.sidebar.button("Database Management", key="database_management_button"):
        st.session_state.page = "Database Management"

display_navbar()

if 'page' not in st.session_state:
    st.session_state.page = "Home"

if st.session_state.page == "Home":
    st.title("WELCOME TO THE SENTIMENT ANALYSIS DASHBOARD!")
    st.markdown("""
        THIS DASHBOARD APPLICATION ALLOWS YOU TO ANALYZE PRODUCT REVIEWS FROM AMAZON USING SENTIMENT ANALYSIS. 
        WITH THIS TOOL, YOU CAN:
        - SCRAPE REVIEWS DIRECTLY FROM AMAZON.
        - UPLOAD YOUR OWN CSV FILES CONTAINING REVIEWS.
        - ANALYZE ANY CUSTOM TEXT TO DETERMINE THE SENTIMENT.
        - VIEW INSIGHTS ABOUT POSITIVE, NEGATIVE, AND NEUTRAL SENTIMENT FOUND IN THE REVIEWS.
    """)

if st.session_state.page == "User Operations":
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("SCRAPE REVIEWS", key="scrape_reviews_button"):
            st.session_state.user_action = "Scrape Reviews"
    with col2:
        if st.button("UPLOAD DATASET", key="upload_dataset_button"):
            st.session_state.user_action = "Upload Dataset"
    with col3:
        if st.button("ANALYZE TEXT", key="analyze_text_button"):
            st.session_state.user_action = "Analyze Text"

    if 'user_action' not in st.session_state:
        st.session_state.user_action = None

    if st.session_state.user_action == "Scrape Reviews":
        st.header("SCRAPE REVIEWS FROM AMAZON")
        url_input = st.text_input("ENTER AMAZON REVIEW URL:")
        pages_input = st.number_input("PAGES TO SCRAPE:", 1, 50, 1)

        if st.button("SCRAPE REVIEWS", key="start_scrape"):
            if url_input:
                with st.spinner('SCRAPING DATA...'):
                    scraped_reviews = scrape_reviews(url_input, pages_input)
                st.success("DATA SCRAPING COMPLETE!")

                df_reviews = pd.DataFrame(scraped_reviews)
                st.write("### SCRAPED REVIEWS")
                st.write(df_reviews)

                if not df_reviews.empty:
                    df_reviews['Processed_Description'] = df_reviews['Description'].apply(preprocess_text)

                    # Analyze sentiment
                    df_reviews['Sentiment'] = df_reviews['Translated_Description'].apply(analyze_sentiment)

                    for _, row in df_reviews.iterrows():
                        insert_scraped_review(row['Name'], row['Rating'], row['Title'], 
                                              row['Description'], row['Sentiment'], 
                                              row['Translated_Description'])

                    st.write("### SENTIMENT DISTRIBUTION")
                    sentiment_counts = df_reviews['Sentiment'].value_counts()
                    sentiment_counts_df = pd.DataFrame(sentiment_counts).reset_index()
                    sentiment_counts_df.columns = ['Sentiment', 'Counts']

                    st.write("#### Pie Chart of Sentiment Distribution")
                    fig_pie = go.Figure(data=[go.Pie(labels=sentiment_counts_df['Sentiment'],
                                                       values=sentiment_counts_df['Counts'],
                                                       hole=0.3,
                                                       marker=dict(colors=['green' if sentiment == 'Positive' 
                                                                           else 'red' if sentiment == 'Negative' 
                                                                           else 'yellow' 
                                                                           for sentiment in sentiment_counts_df['Sentiment']]))])
                    st.plotly_chart(fig_pie)

                    positive_reviews_text = ' '.join(df_reviews[df_reviews['Sentiment'] == 'Positive']['Description'])
                    negative_reviews_text = ' '.join(df_reviews[df_reviews['Sentiment'] == 'Negative']['Description'])

                    with st.container():
                        st.write("### WORD CLOUD FOR REVIEWS")
                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("Positive Reviews Word Cloud")
                            if positive_reviews_text:
                                plt.figure(figsize=(4, 4))
                                wordcloud_pos = WordCloud(width=200, height=200, background_color='black').generate(positive_reviews_text)
                                plt.imshow(wordcloud_pos, interpolation='bilinear')
                                plt.axis('off')
                                st.pyplot(plt)
                                plt.close()
                            else:
                                st.write("**No positive reviews available to generate word cloud.**")
                        
                        with col2:
                            st.subheader("Negative Reviews Word Cloud")
                            if negative_reviews_text:
                                plt.figure(figsize=(4, 4))
                                wordcloud_neg = WordCloud(width=200, height=200, background_color='black').generate(negative_reviews_text)
                                plt.imshow(wordcloud_neg, interpolation='bilinear')
                                plt.axis('off')
                                st.pyplot(plt)
                                plt.close()
                            else:
                                st.write("**No negative reviews available to generate word cloud.**")

                    st.write("### Bar Chart of Ratings by Sentiment")
                    rating_count = df_reviews.groupby(['Sentiment', 'Rating']).size().reset_index(name='Counts')
                    fig_bar = px.bar(rating_count, x='Rating', y='Counts', color='Sentiment', barmode='group',
                                     title='Bar Chart of Ratings by Sentiment', 
                                     color_discrete_map={
                                         'Positive': 'green',
                                         'Negative': 'red',
                                         'Neutral': 'yellow'
                                     },
                                     labels={'Counts': 'Number of Reviews', 'Rating': 'Rating'})
                    st.plotly_chart(fig_bar)

                    insights = generate_insights(df_reviews)
                    st.write("### INSIGHTS")
                    for insight in insights:
                        st.write(insight)

                    # Show the DataFrame with Sentiment as the last column
                    st.write("### SCRAPED REVIEWS WITH SENTIMENT")
                    st.write(df_reviews[['Name', 'Rating', 'Title', 'Description', 'Translated_Description', 'Sentiment']])
                    
                else:
                    st.write("**NO REVIEWS FOUND DURING SCRAPING.**")

    elif st.session_state.user_action == "Upload Dataset":
        st.header("UPLOAD DATASET")
        uploaded_file = st.file_uploader("CHOOSE A CSV FILE", type="csv")
        
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write("### UPLOADED DATA")
            st.write(data)

            required_columns = ['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator', 
                                'HelpfulnessDenominator', 'Score', 'Time', 'Summary', 'Text']
            if all(col in data.columns for col in required_columns):
                data['Processed_Text'] = data['Text'].apply(preprocess_text)
                data['Sentiment'] = data['Processed_Text'].apply(analyze_sentiment)

                for _, row in data.iterrows():
                    insert_uploaded_review(row['ProductId'], row['UserId'], row['ProfileName'], 
                                           row['HelpfulnessNumerator'], row['HelpfulnessDenominator'],
                                           row['Score'], row['Time'], row['Summary'], 
                                           row['Text'], row['Sentiment'])

                st.write("### SENTIMENT DISTRIBUTION")
                sentiment_counts = data['Sentiment'].value_counts()
                sentiment_counts_df = pd.DataFrame(sentiment_counts).reset_index()
                sentiment_counts_df.columns = ['Sentiment', 'Counts']

                st.write("#### Pie Chart of Sentiment Distribution")
                fig_pie = go.Figure(data=[go.Pie(labels=sentiment_counts_df['Sentiment'], 
                                                   values=sentiment_counts_df['Counts'],
                                                   hole=0.3, 
                                                   marker=dict(colors=['green' if sentiment == 'Positive' 
                                                                       else 'red' if sentiment == 'Negative'
                                                                       else 'yellow'
                                                                       for sentiment in sentiment_counts_df['Sentiment']]))])
                st.plotly_chart(fig_pie)

                st.write("### Bar Chart of Scores by Sentiment")
                score_count = data.groupby(['Sentiment', 'Score']).size().reset_index(name='Counts')
                fig_bar = px.bar(score_count, x='Score', y='Counts', color='Sentiment', barmode='group',
                                 title='Bar Chart of Scores by Sentiment', 
                                 color_discrete_map={
                                     'Positive': 'green',
                                     'Negative': 'red',
                                     'Neutral': 'yellow'
                                 },
                                 labels={'Counts': 'Number of Reviews', 'Score': 'Score'})
                st.plotly_chart(fig_bar)

                insights = generate_insights(data)
                st.write("### INSIGHTS")
                for insight in insights:
                    st.write(insight)

                # Display the DataFrame with Sentiment included
                st.write("### UPLOADED REVIEWS WITH SENTIMENT")
                st.write(data[['Id', 'ProductId', 'UserId', 'ProfileName', 
                               'HelpfulnessNumerator', 'HelpfulnessDenominator', 
                               'Score', 'Time', 'Summary', 'Text', 'Sentiment']])
            else:
                st.write("**UPLOADED CSV MUST CONTAIN THE FOLLOWING COLUMNS: Id, ProductId, UserId, ProfileName, HelpfulnessNumerator, HelpfulnessDenominator, Score, Time, Summary, Text.**")

    elif st.session_state.user_action == "Analyze Text":
        st.header("ANALYZE CUSTOM TEXT")
        user_input_text = st.text_area("ENTER TEXT:")
        target_language = st.selectbox("SELECT TARGET LANGUAGE:", ['en', 'es', 'fr', 'de'])
        
        if st.button("TRANSLATE TEXT", key="translate_input"):
            if user_input_text:
                # Translate user input and log the output
                translated_text = translate_text(user_input_text, target_language)
                st.write("### TRANSLATED TEXT")
                st.write(translated_text)

                # Analyze sentiment on the translated text
                sentiment_result = analyze_sentiment(translated_text)
                st.write(f"**SENTIMENT OF TRANSLATED TEXT:** {sentiment_result}")
        
        if st.button("ANALYZE TEXT", key="analyze_input"):
            if user_input_text:
                processed_text = preprocess_text(user_input_text)
                sentiment_result = analyze_sentiment(processed_text)
                st.write(f"**SENTIMENT OF ORIGINAL TEXT:** {sentiment_result}")
            else:
                st.write("**PLEASE ENTER TEXT TO ANALYZE.**")

if st.session_state.page == "Database Management":
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("SHOW ALL SCRAPED REVIEWS", key="show_all_scraped_button"):
            st.session_state.database_action = "Show All Scraped Reviews"
    with col2:
        if st.button("SHOW ALL UPLOADED REVIEWS", key="show_all_uploaded_button"):
            st.session_state.database_action = "Show All Uploaded Reviews"
    with col3:
        if st.button("CLEAR DATABASE", key="clear_database_button"):
            st.session_state.database_action = "Clear Database"

    if 'database_action' not in st.session_state:
        st.session_state.database_action = None

    if st.session_state.database_action == "Show All Scraped Reviews":
        st.header("VIEW ALL SCRAPED REVIEWS")
        if st.button("SHOW REVIEWS", key="display_scraped_reviews_button"):
            reviews = fetch_all_reviews('scraped_reviews')
            if reviews:
                df_reviews = pd.DataFrame(reviews, columns=["ID", "Name", "Rating", "Title", "Description", "Sentiment", "Translated_Description"])
                st.write("### ALL SCRAPED REVIEWS")
                st.write(df_reviews)
            else:
                st.write("**NO SCRAPED REVIEWS FOUND.**")
    
    elif st.session_state.database_action == "Show All Uploaded Reviews":
        st.header("VIEW ALL UPLOADED REVIEWS")
        if st.button("SHOW REVIEWS", key="display_uploaded_reviews_button"):
            reviews = fetch_all_reviews('uploaded_reviews')
            if reviews:
                df_reviews = pd.DataFrame(reviews, columns=["ID", "ProductId", "UserId", "ProfileName", 
                                                             "HelpfulnessNumerator", "HelpfulnessDenominator", 
                                                             "Score", "Time", "Summary", "Text", "Sentiment"])
                st.write("### ALL UPLOADED REVIEWS")
                st.write(df_reviews)
            else:
                st.write("**NO UPLOADED REVIEWS FOUND.**")
    
    elif st.session_state.database_action == "Clear Database":
        st.header("CLEAR ALL REVIEWS FROM BOTH TABLES")
        if st.button("CONFIRM CLEAR", key="confirm_clear_button"):
            clear_database()
