import nltk 
import snscrape.modules.twitter as sntwitter
import pandas as pd
import string
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tqdm import tqdm

# Define the search terms and number of tweets to be fetched
terms = "Elon Musk"
lang = "en"

search_terms = terms + " " + "lang:" + lang

number_of_tweets = 100

# Get tweets data using snscrape library
tweets = sntwitter.TwitterSearchScraper(search_terms).get_items()

# Define a function to process the text in a tweet
def process_text(text):
    # Remove punctuations
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize words
    words = word_tokenize(text)

    # Remove stopwords
    words = [word for word in words if word.lower() not in stopwords.words("english")]

    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Return processed text
    return " ".join(words)

t_list = []

for i, tweet in tqdm(enumerate(tweets)):
    if i >= number_of_tweets:
        break
  
    t_list.append([process_text(tweet.rawContent)])

# Store the tweets in a pandas dataframe
df = pd.DataFrame(t_list, columns = ["text"])

# Define a function to perform sentiment analysis on each tweet using TextBlob library
def get_sentiment(tweet):
    analysis = TextBlob(tweet)
    return analysis.sentiment.polarity



def main():
  # Apply the sentiment analysis function to the dataframe
  df["sentiment"] = df["text"].apply(get_sentiment)

  # Print the sentiment analysis results
  print("\n")
  print(df.head())

if __name__ == "__main__":
  main()
