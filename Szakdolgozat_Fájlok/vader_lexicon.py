import nltk
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
vader_lexicon = sid.lexicon

for word, score in list(vader_lexicon.items())[:1000]:  
    print(word, score)

