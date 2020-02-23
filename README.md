# Twitter Sentiment-Analysis
Dataset being used : 
    [Sentiment 140](https://www.kaggle.com/kazanova/sentiment140)

I. PreProcessing :
    
    1. Remove URL
    2. Remove @
    3. Remove Apostrophe
    4. Remove Newline
    5. Replace Emoticons with suitable text
    6. Remove stopwords
    7. Stemming (PorterStemmer)
    8. Spell Correction
    9. Regex Tokenization
 
II. Data Training :
  
    1. Count Vectorization
    2. LinearSVC - Non Convergent
    3. Gaussian RBF SVC
    4. Polynomial SVC
    5. Decision Trees
 
III. Word Embeddings :
  
    1. Word2Vec
    2. Glove
    3. Count Vectorizer
    4. TF-IDF Vectorizer
    5. Word2Vec with TF-IDF
    6. BERT
    7. BERT with TF-IDF
    8. GPT-2
    
    
 IV. Results :
      
      1. Gaussian RBF
      2. Polynomial SVC
      3. DecisionTree
   
  V. Tests:
      
      1. Confusion Matrix
      2. AUC-ROC
      3. Sklearn accuracy metric
 
