import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
import string
import seaborn as sns
import matplotlib.pyplot as plt

nltk.download('stopwords')
nltk.download('vader_lexicon')

df = pd.read_excel(r'files\\Analysis.xlsx')

analyzer = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()


#-------------------
#FUNCTIONS
#-------------------

def preprocess(text):
    text = text.translate(str.maketrans("","",string.punctuation))
    tokenized_content = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokenized_content if not word in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(token, pos = "v") for token in filtered_tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(token, pos = "n") for token in lemmatized_tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(token, pos = "r") for token in lemmatized_tokens]
    processed_token = ' '.join(lemmatized_tokens)
    return processed_token

def getCompound(text):
    scores = analyzer.polarity_scores(text)
    compound_score = scores['compound']
    return compound_score

def getPositive(text):
    scores = analyzer.polarity_scores(text)
    pos_score = scores['pos']
    return pos_score

def getNegative(text):
    scores = analyzer.polarity_scores(text)
    neg_score = scores['neg']
    return neg_score

def getNeutral(text):
    scores = analyzer.polarity_scores(text)
    neu_score = scores['neu']
    return neu_score

def getSubjectivity(text):
    subjectivity_score = TextBlob(text).subjectivity
    return subjectivity_score

def getTextBlob(text):
    polarity_score = TextBlob(text).polarity
    return polarity_score

#-------------------
#MAIN PROCESS
#-------------------

df['Cleaned Analysis'] = df['Analysis'].apply(preprocess)
df['VADER Compound'] = df['Cleaned Analysis'].apply(getCompound)
df['VADER Positive'] = df['Cleaned Analysis'].apply(getPositive)
df['VADER Negative'] = df['Cleaned Analysis'].apply(getNegative)
df['VADER Neutral'] = df['Cleaned Analysis'].apply(getNeutral)
df['TextBlob Subjectivity'] = df['Cleaned Analysis'].apply(getSubjectivity)
df['TextBlob Polarity'] = df['Cleaned Analysis'].apply(getTextBlob)
df['Weights VADER Compound'] = round(df['VADER Compound'].div(df['VADER Compound'].sum()), 5)
df['Weights VADER Positive'] = round(df['VADER Positive'].div(df['VADER Positive'].sum()), 5)
df['Weights TextBlob Polarity'] = round(df['TextBlob Polarity'].div(df['TextBlob Polarity'].sum()), 5)

df.to_csv(r'files\\Sentiment_weights.csv')

