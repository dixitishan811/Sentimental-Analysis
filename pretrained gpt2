import nltk.sentiment.vader as sent
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn import metrics
import emoji
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim.models import Word2Vec
from textblob import TextBlob
import re
from nltk.tokenize import regexp_tokenize
from bert_embedding import BertEmbedding
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel
import time
import winsound


class Data:
    def __init__(self):
        self.analyzer = sent.SentimentIntensityAnalyzer()

        self.raw_tweets = None

        self.train_labels = None
        self.test_labels = None

        self.nos_train_neu = 0
        self.nos_train_pos = 0
        self.nos_train_neg = 0

        self.nos_test_neu = 0
        self.nos_test_neg = 0
        self.nos_test_pos = 0

        self.get_tweets()

    def get_tweets(self):
        filename = 'Sentiment140.csv'
        raw_train = pd.read_csv(filename,
                                header=None,
                                names=['Score', 'A', 'B', 'C', 'D', 'Tweet'])
        self.raw_tweets = pd.Series(raw_train['Tweet'], dtype='str')

    def create_train_label(self, train_tweet_sequence):
        score = []
        for tweet in train_tweet_sequence:
            val = round(self.analyzer.polarity_scores(tweet)['compound'])
            if val == 0:
                self.nos_train_neu += 1
            elif val == 1:
                self.nos_train_pos += 1
            else:
                self.nos_train_neg += 1
            score.append(val)
        self.train_labels = pd.Series(score)

    def create_test_label(self, test_tweet_sequence):
        score = []
        for tweet in test_tweet_sequence:
            val = round(self.analyzer.polarity_scores(tweet)['compound'])
            if val == 0:
                self.nos_test_neu += 1
            elif val == 1:
                self.nos_test_pos += 1
            else:
                self.nos_test_neg += 1
            score.append(val)
        self.test_labels = pd.Series(score)


class PreProcessing:
    def __init__(self, data_obj, processing_size=1600000, stem_and_stop=False):
        data_obj.raw_tweets = data_obj.raw_tweets[: processing_size]

        self.detokenized_corpus = []
        self.tokenized_corpus = []

        self.remove_url(data_obj)
        self.remove_at(data_obj)
        self.remove_apostrophe(data_obj)
        self.remove_new_line(data_obj)
        self.replace_emoticons(data_obj)
        if stem_and_stop:
            self.remove_stopwords_and_stem(data_obj)
        self.spell_correction(data_obj, stem_and_stop)

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
        for i in range(len(data_ob.raw_tweets)):
            data_ob.raw_tweets[i] = emoji.demojize(data_ob.raw_tweets[i])
        data_ob.raw_tweets = data_ob.raw_tweets.str.replace("_", ' ', case=False)

    def remove_stopwords_and_stem(self, data_ob):
        my_stopwords = list(stopwords.words('english'))
        ps = PorterStemmer()
        for tweet in data_ob.raw_tweets:
            temp_list = [ps.stem(word) for word in TreebankWordTokenizer().tokenize(tweet) if word not in my_stopwords]
            self.tokenized_corpus.append(temp_list)

    def spell_correction(self, data_obj, stem_and_stop):
        if not stem_and_stop:
            self.tokenized_corpus = [[word for word in tweet.split()]for tweet in data_obj.raw_tweets]

        self.tokenized_corpus = [[str(TextBlob(word).correct()) for word in tweet] for tweet in self.tokenized_corpus]
        self.detokenized_corpus = [TreebankWordDetokenizer().detokenize(tweet) for tweet in self.tokenized_corpus]

        self.detokenized_corpus = [re.sub('[^A-Za-z0-9 ]', '', tweet) for tweet in self.detokenized_corpus]
        self.tokenized_corpus = [regexp_tokenize(tweet, r'\S*') for tweet in self.detokenized_corpus]


class WordEmbedding:
    def __init__(self, prep_obj, method='w2v_tfidf'):
        self.vector_corpus = []

        if method == 'w2v_tfidf':
            self.word_to_vec(prep_obj, use_tf_idf=True)
        elif method == 'w2v':
            self.word_to_vec(prep_obj, use_tf_idf=False)
        elif method == 'cv':
            self.count_vectorizer(prep_obj)
        elif method == 'tfidf':
            self.tf_idf(prep_obj)
        elif method == 'glove':
            self.glove(prep_obj)
        elif method == 'bert':
            self.bert(prep_obj, use_tf_idf=False)
        elif method == 'bert_tf_idf':
            self.bert(prep_obj, use_tf_idf=True)
        elif method == 'gpt2':
            self.gpt2(prep_obj)

    def bert(self, prep_obj, use_tf_idf=False):
        self.tf_idf(prep_obj)
        tf_idf_vector = self.vector_corpus
        self.vector_corpus = []

        bert_embedding = BertEmbedding(model='bert_12_768_12', dataset_name='book_corpus_wiki_en_cased')
        result = np.array(bert_embedding(prep_obj.detokenized_corpus))

        vec = np.zeros(768)
        for row, sentence in enumerate(result):
            for column, word_vec in enumerate(sentence[1]):
                mul = tf_idf_vector[row][column] if use_tf_idf else 1
                vec = np.add(vec, np.array(word_vec) * mul)
            vec = np.true_divide(vec, 1 if len(sentence[1]) == 0 else len(sentence[1]))
            self.vector_corpus.append(vec)

    def gpt2(self, prep_obj):
        self.vector_corpus = []

        model = GPT2LMHeadModel.from_pretrained('gpt2')
        token_maker = GPT2Tokenizer.from_pretrained('gpt2')
        for tweet in prep_obj.detokenized_corpus:
            text_index = token_maker.encode(tweet)
            vector = (model.transformer.wte.weight[text_index, :])
            vector = vector.detach().numpy()
            vector = np.sum(vector, axis=0)
            self.vector_corpus.append(vector)

    def count_vectorizer(self, prep_obj):
        self.vector_corpus = CountVectorizer().fit_transform(prep_obj.detokenized_corpus)

    def tf_idf(self, prep_obj):
        vectorizer = TfidfVectorizer(lowercase=False, analyzer='word', token_pattern="\S*")
        vec = vectorizer.fit_transform(prep_obj.detokenized_corpus).todense().tolist()
        feature_names = vectorizer.get_feature_names()
        self.vector_corpus = [[vec[index][feature_names.index(word)] for word in tweet]
                              for index, tweet in enumerate(prep_obj.tokenized_corpus)]

    def word_to_vec(self, prep_obj, use_tf_idf=False):
        self.tf_idf(prep_obj)
        tf_idf_vector = self.vector_corpus
        self.vector_corpus = []

        features = 100
        model = gensim.models.Word2Vec(prep_obj.tokenized_corpus, min_count=1, workers=4, size=features, window=5, sg=0)
        vec = np.zeros(features)
        for row, tweet in enumerate(prep_obj.tokenized_corpus):
            for column, word in enumerate(tweet):
                mul = tf_idf_vector[row][column] if use_tf_idf else 1
                vec = np.add(vec, np.array(model.wv[word]) * mul)
            vec = np.true_divide(vec, 1 if len(tweet) == 0 else len(tweet))
            self.vector_corpus.append(vec)

    def glove(self, prep_obj):
        embeddings_dict = {}
        features = 100
        with open("glove.twitter.27B." + str(features) + "d.txt", 'r', encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector

        # if word is not in dictionary then its corresponding vector is taken as zero
        vec = np.zeros(features)
        self.vector_corpus = []
        for tweet in prep_obj.tokenized_corpus:
            for word in tweet:
                try:
                    vec = np.add(vec, np.array(embeddings_dict[word]))
                except ValueError:
                    pass
            vec = np.true_divide(vec, 1 if len(tweet) == 0 else len(tweet))
            self.vector_corpus.append(vec)


class Training:
    def __init__(self, data_obj, prep_obj, word_embed_obj, train_size=80000, method='dt'):
        data_obj.create_train_label(prep_obj.detokenized_corpus[: train_size])

        self.X_train = word_embed_obj.vector_corpus[: train_size]
        self.Y_train = data_obj.train_labels

        self.clf = None

        if method == 'dt':
            self.decision_tree()
        elif method == 'lin_svc':
            self.linear_svc()
        elif method == 'poly_svc':
            self.poly_svc()
        elif method == 'gauss_svc':
            self.gaussian_svc()

    def linear_svc(self):
        self.clf = LinearSVC()
        self.clf.fit(self.X_train, self.Y_train)

    def poly_svc(self):
        self.clf = SVC(kernel='poly', degree=3, coef0=1, C=100)
        self.clf.fit(self.X_train, self.Y_train)

    def gaussian_svc(self):
        self.clf = SVC(kernel='rbf', gamma=5, C=100)
        self.clf.fit(self.X_train, self.Y_train)

    def decision_tree(self):
        self.clf = tree.DecisionTreeClassifier()
        self.clf.fit(self.X_train, self.Y_train)


class Testing:
    def __init__(self, data_obj, prep_obj, word_embed_obj, training_obj, test_size=20000):
        train_size = len(data_obj.train_labels)

        data_obj.create_test_label(prep_obj.detokenized_corpus[train_size: train_size + test_size])

        self.X_test = word_embed_obj.vector_corpus[train_size:  train_size + test_size]
        self.Y_test = data_obj.test_labels

        self.clf = training_obj.clf
        self.Y_pred = None

        self.accuracy()
        self.auc()
        self.confusion_matrix()

    def accuracy(self):
        self.Y_pred = self.clf.predict(self.X_test)
        print('Accuracy is :', accuracy_score(self.Y_test, self.Y_pred) * 100)

    def auc(self):
        print("\nArea under curve :")
        self.Y_pred = self.clf.predict(self.X_test)
        fpr, tpr, thresholds = metrics.roc_curve(self.Y_test, self.Y_pred, pos_label=1)
        print("1", metrics.auc(fpr, tpr))
        fpr, tpr, thresholds = metrics.roc_curve(self.Y_test, self.Y_pred, pos_label=0)
        print("0", metrics.auc(fpr, tpr))
        fpr, tpr, thresholds = metrics.roc_curve(self.Y_test, self.Y_pred, pos_label=-1)
        print("-1", metrics.auc(fpr, tpr))

    def confusion_matrix(self):
        print('\nConfusion Matrix :')
        self.Y_pred = self.clf.predict(self.X_test)
        print(confusion_matrix(self.Y_test, self.Y_pred))


processing_size = 1000
train_fraction = 0.8

train_size = int(train_fraction * processing_size)
test_size = int((1 - train_fraction) * processing_size)

start_time = time.process_time()

dt = Data()
pp = PreProcessing(dt, processing_size=processing_size, stem_and_stop=False)
we = WordEmbedding(pp, method='gpt2')
tr = Training(dt, pp, we, train_size=train_size, method='dt')
te = Testing(dt, pp, we, tr, test_size=test_size)

end_time = time.process_time()

print("\nTime taken =", end_time - start_time)
print('\nnos_test_neg =', dt.nos_test_neg, '\tnos_train_neg =', dt.nos_train_neg)
print('nos_test_neu =', dt.nos_test_neu, '\tnos_train_neu =', dt.nos_train_neu)
print('nos_test_pos =', dt.nos_test_pos, '\tnos_train_pos =', dt.nos_train_pos)
winsound.Beep(3000, 2000)
