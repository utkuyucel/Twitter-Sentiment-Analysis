import nltk 
import snscrape.modules.twitter as sntwitter
import pandas as pd
import string
import re
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
import enchant
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta    
from wordcloud import WordCloud
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.sentiment.vader import SentimentIntensityAnalyzer



#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('maxent_ne_chunker')
#nltk.download('words')
#nltk.download('omw-1.4')
#nltk.download('vader_lexicon')

def search_tweets(terms, lang, number_of_tweets):
    """
    Fetches tweets containing the specified search terms in the specified language, within a given date range.
    Returns a list of raw tweet contents.
    """
    tweets = sntwitter.TwitterSearchScraper(terms).get_items()
    raw_tweets = []
    for i, tweet in tqdm(enumerate(tweets), total=number_of_tweets):
        if i >= number_of_tweets:
            break
        raw_tweets.append(tweet.content)
    return raw_tweets

def process_text(text):
    """
    Cleans the text of a tweet by removing punctuation, stop words, and lemmatizing words.
    Identifies and removes named entities and corrects misspelled words.
    Returns the cleaned text as a string.
    """
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    
    # Remove punctuation and convert to lowercase
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    
    # Tokenize text and tag parts of speech
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    
    print("Identifying name entities...\n")
    # Identify named entities and remove them from text
    named_entities = ne_chunk(pos_tags, binary=True)
    named_entities = ["/".join(word for word, tag in elt) for elt in tqdm(named_entities) if isinstance(elt, nltk.tree.Tree)]
    for named_entity in named_entities:
        text = text.replace(named_entity, "")
    
    print("Correting spelling errors...\n")
    # Correct spelling errors
    spell_checker = enchant.Dict("en_US")
    words = [word if spell_checker.check(word) else (spell_checker.suggest(word)[0] if spell_checker.suggest(word) else word) for word in tqdm(words)]
    
    # Remove stop words and lemmatize words
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

def get_sentiment_bert(tweet):
    """
    Performs sentiment analysis on the text of a tweet using a pre-trained BERT model from Hugging Face Transformers.
    Returns the predicted sentiment as a string: "positive", "negative", or "neutral".
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

    # Tokenize input text and convert to PyTorch tensors
    inputs = tokenizer.encode_plus(tweet, return_tensors="pt", truncation=True, padding=True)

    # Pass input through the model and get predicted label
    outputs = model(**inputs)
    predicted_label = torch.argmax(outputs.logits).item()

    # Map predicted label to sentiment string
    if predicted_label == 0:
        return "negative"
    elif predicted_label == 1:
        return "positive"
    else:
        return "neutral"

def get_sentiment_vader(tweet):
    """
    Performs sentiment analysis on the text of a tweet using VADER (Valence Aware Dictionary and sEntiment Reasoner).
    Returns the compound score, which ranges from -1 (most negative) to 1 (most positive).
    """
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(tweet)
    return scores["compound"]


def visualize_sentiment(df, title = "Sentiment"):
    """
    Generates a histogram of sentiment scores for the tweets in the dataframe.
    """
    fig, ax = plt.subplots()
    plt.figure(figsize = (15,15))
    ax.hist(df["sentiment"], bins=20)
    ax.set_xlabel("Sentiment Score")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    plt.show()

def main():
    # Set search terms, language, and number of tweets to fetch
    terms = "Bitcoin OR #BTC OR $BTC"
    lang = "en"
    number_of_tweets = 300

    # Fetch tweets
    raw_tweets = search_tweets(terms, lang, number_of_tweets)

    print("Cleaning and processing...\n")
    # Clean and process tweets
    processed_tweets = [process_text(tweet) for tweet in tqdm(raw_tweets)]

    print("Performing analysis...\n")
    # Perform sentiment analysis on tweets
    sentiment_scores = [get_sentiment_vader(tweet) for tweet in tqdm(processed_tweets)]

    # Store the tweets, sentiment scores, and timestamps in a pandas dataframe
    df = pd.DataFrame({"text": processed_tweets, "sentiment": sentiment_scores, "timestamp": pd.Timestamp.now()})

    print("Visualizing...\n")
    # Visualize sentiment scores
    visualize_sentiment(df, "Sentiment")

    print("\n")

    # Print out tweets with highest and lowest sentiment scores
    df_sorted = df.sort_values("sentiment")
    print("Most negative tweets:")
    for tweet in df_sorted["text"].head(5):
        print("-", tweet)
    print("\nMost positive tweets:")
    for tweet in df_sorted["text"].tail(5):
        print("-", tweet)


    print("\n")

    # Generate word cloud
    words = Counter(" ".join(processed_tweets).split()).most_common(100)
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(dict(words))
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    
    


if __name__ == "__main__":
  main()
