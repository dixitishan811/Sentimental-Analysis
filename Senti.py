import nltk.sentiment.vader as senti
import pandas as pd
import emoji
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

raw_train = pd.read_csv('training.1600000.processed.noemoticon.csv', header=None,
                        names=['Score', 'A', 'B', 'C', 'D', 'Tweet'])

raw_train.drop(['A', 'B', 'C', 'D'], axis=1)

analyzer = senti.SentimentIntensityAnalyzer()
raw_train = pd.Series(raw_train['Tweet'], dtype='str')

score_train = []

train = raw_train[:80000]
test = raw_train[80000:100000]

for i in range(80000):
    score_train.append(analyzer.polarity_scores(train[i])['compound'])

pd.Series(score_train)
# train = {'Tweets' : train,'Score' : score_train }
# train = pd.DataFrame(train)

score_test = []

for i in range(80000, 100000):
    score_test.append(analyzer.polarity_scores(test[i])['compound'])

pd.Series(score_test)
# test = {'Tweets' : test,'Score' : score_test }
# test = pd.DataFrame(test)


raw_train = raw_train.str.replace('http\S+|www.\S+', '', case=False)
raw_train = raw_train.str.replace('@\S+', '', case=False)
raw_train = raw_train.str.replace("’|'\S+", '', case=False)
raw_train = raw_train.str.replace("\\n", '', case=False)
for i in range(1600000):
    raw_train[i] = emoji.demojize(raw_train[i])

raw_train = raw_train.str.replace("_", ' ', case=False)

my_stopwords = list(stopwords.words('english'))
ps = PorterStemmer()
temp_list = []
temp_temp_list = []
for i in range(160000):
    temp_temp_list.append(TreebankWordTokenizer().tokenize(raw_train[i]))
    for index, word in enumerate(temp_temp_list):
        if word in my_stopwords:
            temp_temp_list.remove(word)
        temp_temp_list[index] = ps.stem(word)

    temp_list.append(TreebankWordDetokenizer().detokenize(temp_temp_list))


print(raw_train[513641])