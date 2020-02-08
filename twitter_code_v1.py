import nltk.sentiment.vader as sent
import pandas as pd
import emoji
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


class Data:
    def __init__(self):
        self.analyzer = sent.SentimentIntensityAnalyzer()

        self.raw_tweets = None
        self.get_tweets()

        self.train_tweets = self.raw_tweets[:80000]
        self.test_tweets = self.raw_tweets[80000: 100000]

        self.train_labels = None
        self.test_labels = None
        self.create_label(self.train_tweets, self.test_tweets)

    def get_tweets(self):
        raw_train = pd.read_csv('Sentiment140.csv', header=None, names=['Score', 'A', 'B', 'C', 'D', 'Tweet'])
        self.raw_tweets = pd.Series(raw_train['Tweet'], dtype='str')

    def create_label(self, train_tweet_sequence, test_tweet_sequence):
        score = []
        for tweet in train_tweet_sequence:
            score.append(round(self.analyzer.polarity_scores(tweet)['compound']))
        self.train_labels = pd.Series(score)

        score = []
        for tweet in test_tweet_sequence:
            score.append(round(self.analyzer.polarity_scores(tweet)['compound']))
        self.test_labels = pd.Series(score)


class PreProcessing:
    def __init__(self, data_obj):
        self.my_stopwords = list(stopwords.words('english'))
        self.ps = PorterStemmer()
        self.corpus = []
        self.temp_list = []

        self.remove_url(data_obj)
        self.remove_at(data_obj)
        self.remove_apostrophe(data_obj)
        self.remove_new_line(data_obj)
        self.replace_emoticons(data_obj)
        self.remove_stopwords_and_stem(data_obj)

    @staticmethod
    def remove_url(data_ob):
        data_ob.raw_tweets = data_ob.raw_tweets.str.replace('http\S+|www.\S+', '', case=False)

    @staticmethod
    def remove_at(data_ob):
        data_ob.raw_tweets = data_ob.raw_tweets.str.replace('@\S+', '', case=False)

    @staticmethod
    def remove_apostrophe(data_ob):
        data_ob.raw_tweets = data_ob.raw_tweets.str.replace("’|'\S+", '', case=False)

    @staticmethod
    def remove_new_line(data_ob):
        data_ob.raw_tweets = data_ob.raw_tweets.str.replace("\\n", '', case=False)

    @staticmethod
    def replace_emoticons(data_ob):
        data_ob.raw_tweets = pd.Series([emoji.demojize(sentence) for sentence in data_ob.raw_tweets], dtype='str')
        data_ob.raw_tweets = data_ob.raw_tweets.str.replace("_", ' ', case=False)

    def remove_stopwords_and_stem(self, data_ob):
        for tweet in data_ob.raw_tweets:
            self.temp_list = TreebankWordTokenizer().tokenize(tweet)
            for index, word in enumerate(self.temp_list):
                word = self.ps.stem(word)
                self.temp_list[index] = word
                if word in self.my_stopwords:
                    self.temp_list.remove(word)

            self.corpus.append(TreebankWordDetokenizer().detokenize(self.temp_list))


x = Data()
y = PreProcessing(x)
