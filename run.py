import nltk 

#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('omw-1.4')

import snscrape.modules.twitter as sntwitter
import pandas as pd
import string
import re
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tqdm import tqdm
import matplotlib.pyplot as plt

def search_tweets(terms, lang, number_of_tweets):
    """
    Fetches tweets containing the specified search terms in the specified language.
    Returns a list of raw tweet contents.
    """
    search_terms = terms + " lang:" + lang
    tweets = sntwitter.TwitterSearchScraper(search_terms).get_items()
    raw_tweets = []
    for i, tweet in tqdm(enumerate(tweets), total=number_of_tweets):
        if i >= number_of_tweets:
            break
        raw_tweets.append(tweet.content)
    return raw_tweets

def process_text(text):
    """
    Cleans the text of a tweet by removing punctuation, stop words, and lemmatizing words.
    Returns the cleaned text as a string.
    """
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = word_tokenize(text)
    words = [word for word in words if word.lower() not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

def get_sentiment(tweet):
    """
    Performs sentiment analysis on the text of a tweet using TextBlob library.
    Returns the polarity of the sentiment as a float.
    """
    analysis = TextBlob(tweet)
    return analysis.sentiment.polarity

def visualize_sentiment(df):
    """
    Generates a histogram of sentiment scores for the tweets in the dataframe.
    """
    fig, ax = plt.subplots()
    plt.figure(figsize = (15,15))
    ax.hist(df["sentiment"], bins=20)
    ax.set_xlabel("Sentiment Score")
    ax.set_ylabel("Frequency")
    plt.show()

def main():
    terms = "Bitcoin"
    lang = "en"
    number_of_tweets = 5000
    
    # Fetch tweets
    raw_tweets = search_tweets(terms, lang, number_of_tweets)
    
    # Clean and process tweets
    processed_tweets = [process_text(tweet) for tweet in raw_tweets]
    
    # Perform sentiment analysis on tweets
    sentiment_scores = [get_sentiment(tweet) for tweet in processed_tweets]
    
    # Store the tweets and sentiment scores in a pandas dataframe
    df = pd.DataFrame({"text": processed_tweets, "sentiment": sentiment_scores})
    
    # Visualize sentiment scores
    visualize_sentiment(df)

    print(df.head())

if __name__ == "__main__":
    main()

