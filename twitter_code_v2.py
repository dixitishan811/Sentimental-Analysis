import nltk.sentiment.vader as sent
import pandas as pd
import emoji
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

class Data:
    def __init__(self):
        self.analyzer = sent.SentimentIntensityAnalyzer()

        self.raw_tweets = None
        self.get_tweets()
        self.raw_tweets = self.raw_tweets[:10000]

        self.train_labels = None
        self.test_labels = None
        # self.create_label(self.train_tweets, self.test_tweets)

    def get_tweets(self):
        raw_train = pd.read_csv('training.1600000.processed.noemoticon.csv', header=None, names=['Score', 'A', 'B', 'C', 'D', 'Tweet'])
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
    def hyphen_to_space(data_ob):

        data_ob.raw_tweets = data_ob.raw_tweets.str.replace("-", ' ', case=False)

    @staticmethod
    def replace_emoticons(data_ob):

        # data_ob.raw_tweets = pd.Series([emoji.demojize(sentence) for sentence in data_ob.raw_tweets], dtype='str')
        for i in range(len(data_ob.raw_tweets)):
            data_ob.raw_tweets[i] = emoji.demojize(data_ob.raw_tweets[i])
        data_ob.raw_tweets = data_ob.raw_tweets.str.replace("_", ' ', case=False)

    def remove_stopwords_and_stem(self, data_ob):

        for tweet in data_ob.raw_tweets:
            self.temp_list = []
            for word in TreebankWordTokenizer().tokenize(tweet):
                if word not in self.my_stopwords:
                    self.temp_list.append(self.ps.stem(word))

            self.corpus.append(TreebankWordDetokenizer().detokenize(self.temp_list))


class Training :
    def __init__(self, prep_obj, data_obj):
        self.vectorizer = CountVectorizer()
        self.vector_corpus = self.vectorizer.fit_transform(prep_obj.corpus[:10000]).toarray()
        self.X_test = self.vector_corpus[8000:10000]
        self.X = self.vector_corpus[:8000]
        self.Y = data_obj.train_labels
        self.clf = None
        self.y_true = data_obj.test_labels
        self.y_pred = None


    def linear_SVC(self):
        self.clf = LinearSVC()
        self.clf.fit(self.X, self.Y)

    def gaussian_SVC(self):
        self.clf = SVC(kernel='rbf',gamma=5,C=0.01)
        self.clf.fit(self.X, self.Y)

    def accuracy(self):
        self.y_pred = self.clf.predict(self.X_test)
        print(accuracy_score(self.y_true, self.y_pred))

x = Data()
y = PreProcessing(x)
x.create_label(y.corpus[: 8000], y.corpus[8000: 10000])
z = Training(y, x)
z.gaussian_SVC()
z.accuracy()


