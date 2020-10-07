import re
import string
import emoji
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from itertools import chain
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, f1_score, multilabel_confusion_matrix
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, FunctionTransformer

### Setting important options
pd.options.display.max_columns = None
%matplotlib inline
sns.set(style="darkgrid")
random_seed = 42

# importing tweets from csv and extracting link
tweets = pd.read_csv('tweets.csv', lineterminator='\n', index_col=0, names=['text'], skiprows=1)
tweets.head(5)


class TextCleaner:
    """
    Perform some cleaning on the text data in order to prepare
    it for handling in our machine learning model.
    """

    def __init__(self, links=False, mentions=False, remove_stopwords=False, remove_punctuation=False, remove_emoji=True,
                 lemmatize=False, uncapitalize=False, tokenize=False, split_candidates=False):

        self.links = links
        self.mentions = mentions
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.remove_emoji = remove_emoji
        self.lemmatize = lemmatize
        self.uncapitalize = uncapitalize
        self.tokenize = tokenize
        self.split_candidates = split_candidates

        if self.mentions:
            # Create set of mentions to keep
            self.include = set(['@realDonaldTrump', '@HillaryClinton', '@BarackObama'])

        if self.remove_stopwords:
            # Create a set of stopwords
            self.stop = set(stopwords.words('english'))

        if self.remove_punctuation:
            # Create a set of punctuation words
            self.exclude = set(string.punctuation)
            self.exclude.add('\n')

        if self.remove_emoji:
            # Create a set of emojies
            self.emoji = set(emoji.UNICODE_EMOJI)

        if self.lemmatize:
            # This is the function making the lemmatization
            self.lemma = WordNetLemmatizer()

        if self.tokenize:
            # This is the function performing the tokenization
            self.tokenizer = RegexpTokenizer(r'\w+')

        if self.split_candidates:
            # This is the function splitting text into candidates
            self.don_tags = ['trump', 'donaldtrump', 'realdonaldtrump', 'maga', 'neverhillary', 'trumppence16',
                             'crookedhillary']
            self.hil_tags = ['hillary', 'hillaryclinton', 'nevertrump', 'imwithher', 'dumptrump']

    def clean(self, text):
        """
        Perform cleaning operations to the input pd.Series
        :param text: the string to perform cleaning on

        Return:
            cleaned pd.Dataframe
        """

        if not isinstance(text, str):
            try:
                text = str(text)
            except Exception as e:
                raise ValueError('ValueError: Cannot convert input to String, Exception: {}.'.format(e))
                return

        if self.links:
            # A function that removes the hyperlinks from the text
            text = re.sub(r'http\S+', '', text)

        if self.mentions:
            # A function that removes the mentions from the text
            regex = r'@[^:\s]+'
            mentions = re.findall(regex, text)
            mentions = ' '.join(mention for mention in mentions if mention in self.include)
            text = ' '.join([re.sub(regex, '', text), mentions])

        if self.remove_stopwords:
            # Remove stopwords from text
            text = ' '.join([i for i in text.lower().split() if i not in self.stop])

        if self.remove_punctuation:
            # Remove punctuation from text
            text = ''.join(ch for ch in text if ch not in self.exclude)

        if self.remove_emoji:
            # Remove emojies from text
            text = ''.join(word for word in text if word not in self.emoji)

        if self.lemmatize:
            # Lemmatize text
            text = ' '.join(self.lemma.lemmatize(word) for word in text.split())

        if self.uncapitalize:
            # Remove any capatilization
            text = text.lower()

        return text

    def extract_links(self, text):
        """
        Extract any hyperlinks from input text
        :param text: the string to perform link extraction on

        Return:
            str(link)
        """
        if not isinstance(text, str):
            try:
                text = str(text)
            except Exception as e:
                raise ValueError('ValueError: Cannot convert input to String, Exception: {}.'.format(e))
                return

        if self.links:
            # A function that extracts the hyperlinks from the tweet's content.
            regex = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
            match = re.search(regex, text)
            link = match.group() if match else ''

        return link

    def extract_emojies(self, text):
        """
        Extract any emojies from input text
        :param text: the string to perform emoji extraction on

        Return:
            list of emojies
        """

        if self.remove_emoji:
            # A function that extracts emojies from the tweet's content.
            emojies = ''.join(emoji for emoji in text if emoji in self.emoji)

        return emojies

    def get_tokens(self, text):
        """
        Extract all words from input text
        :param text: the string to perform tokenization on

        Return:
            list of words from text
        """
        text = self.tokenizer.tokenize(text) if self.tokenize else ''

        return text

    def candidate_splitter(self, tokens):
        """
        Check if any word in tokens is concerning any of the candidates
        :param tokens: the list of tokens to check

        Return:
            boolean (x, y) with x=True if tokens contain trump tags and y=True if tokens contain hillary tags
        """
        contains_trump = any(x in tokens for x in self.don_tags)
        contains_hillary = any(x in tokens for x in self.hil_tags)

        return contains_trump, contains_hillary


print(type(tweets))
tweets.head(5)

cleaner = TextCleaner(links=True, mentions=True, remove_stopwords=True, remove_punctuation=True, lemmatize=True, uncapitalize=True,                         tokenize=True, split_candidates=True)

cleaned_tweets = tweets.copy()
cleaned_tweets['text'] = tweets['text'].apply(lambda tweet: cleaner.clean(tweet))
cleaned_tweets['link'] = tweets['text'].apply(lambda tweet: cleaner.extract_links(tweet))
cleaned_tweets['emojies'] = tweets['text'].apply(lambda tweet: cleaner.extract_emojies(tweet))

cleaned_tweets['tokens'] = cleaned_tweets['text'].apply(lambda tweet: cleaner.get_tokens(tweet))
cleaned_tweets['about_trump'] = cleaned_tweets['tokens'].apply(lambda tweet: cleaner.candidate_splitter(tweet)[0])
cleaned_tweets['about_hillary'] = cleaned_tweets['tokens'].apply(lambda tweet: cleaner.candidate_splitter(tweet)[1])
cleaned_tweets.head(5)

cleaned_tweets.shape

cleaned_tweets.to_csv('cleaned_tweets.csv')

[line for line in cleaned_tweets['text']][:20]

cleaned_tweets['link'].loc[cleaned_tweets['link'] != '']


"""Training the model on airline dataset"""
# Now, let's assemble our training dataset
airline_tweets = pd.read_csv('airline_tweets.csv')
airline_tweets = airline_tweets[['airline_sentiment', 'airline_sentiment_confidence', 'text']]
airline_tweets.head()

airline_tweets.shape

# and perform the same cleaning operations as above:
airline_tweets['text'] = airline_tweets['text'].apply(lambda tweet: cleaner.clean(tweet))
airline_tweets.head()


data = airline_tweets.copy()
data = data.rename(columns={"airline_sentiment": "class", "airline_sentiment_confidence": "confidence"})
data.head(3)

data.shape

data['class'].value_counts()


# setup model pipeline
pipeline = Pipeline([
    ('cv', CountVectorizer(lowercase=False)),
    ('tfidf', TfidfTransformer(use_idf=False, smooth_idf=False, sublinear_tf=True)),
    #('ft', FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)),
    #('pw', PowerTransformer(method='yeo-johnson')),
    ('clf', MultinomialNB(alpha=0.02))
])

# train the model using KFold validation
kf = KFold(n_splits=10)
scores = []
cnf_matrix = np.zeros((3, 3))

for train_ind, test_ind in kf.split(data['text'].values):
    train_txt = data.iloc[train_ind]['text'].values
    train_y = data.iloc[train_ind]['class'].values

    test_txt = data.iloc[test_ind]['text'].values
    test_y = data.iloc[test_ind]['class'].values

    pipeline.fit(train_txt, train_y)
    y_proba = pipeline.predict_proba(test_txt)
    y_pred = pipeline.predict(test_txt)
    max_prob = np.amax(y_proba, axis=1)
    select = np.where(max_prob > 0.8, True, False)

    y_pred_filtered = y_pred[select]
    test_y_filtered = test_y[select]

    cnf_matrix += confusion_matrix(test_y, y_pred)
    score = f1_score(test_y, y_pred, pos_label='positive', average='weighted')
    scores.append(score)

print('Total documents fitted: {}'.format(len(data)))
print('Total documents that passed treshhold: {}'.format(int(np.sum(cnf_matrix))))
print('Average accuracy: {}'.format(np.round(np.mean(scores),4)))
print('Confusion matrix:')
print(cnf_matrix)

class_names=pipeline.classes_ # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

plt.show()

# now train our model on the full dataset
all_train = data.iloc[train_ind]['text'].values
all_y = data.iloc[train_ind]['class'].values

clf = pipeline.fit(all_train, all_y)

tweets_text = cleaned_tweets['text']
prob = np.amax(clf.predict_proba(tweets_text), axis=1)
predicted = clf.predict(tweets_text)
predicted.shape

cleaned_tweets['sentiment'] = predicted
cleaned_tweets['confidence'] = prob

cleaned_tweets.head()
cleaned_tweets['sentiment'].value_counts()
cleaned_tweets.shape
cleaned_tweets.to_csv('tweets_with_sentiment.csv')