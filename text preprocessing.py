#import  libraries
import os
# Supress Warnings
import warnings
warnings.filterwarnings('ignore')

#loading all necessary libraries
import numpy as np
import pandas as pd
import nltk
import pandas as pd
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import nltk
nltk.download('omw-1.4')

import string
import matplotlib.pyplot as plt

#importing data
os.chdir(r"C:\Users\User\Downloads")
df=pd.read_csv("TCS_feedback.csv")
df
df.head()

lemma = WordNetLemmatizer()
stop_words = stopwords.words('english')
lemma
stop_words

def text_prep(x):
     corp = str(x).lower()
     corp = re.sub('[^a-zA-Z]+',' ', corp).strip()
     tokens = word_tokenize(corp)
     words = [t for t in tokens if t not in stop_words]
     lemmatize = [lemma.lemmatize(w) for w in words]

     return lemmatize
 
import nltk
nltk.download('punkt')
nltk.download('wordnet')
#for pros col
preprocess_tag = [text_prep(i) for i in df['Pros']]
df["preprocess_txt"] = preprocess_tag
preprocess_tag
df["preprocess_txt"]

df
#for cons col
preprocess_tag = [text_prep(i) for i in df['Cons']]
df["preprocess_txt"] = preprocess_tag
preprocess_tag
df["preprocess_txt"]


#sentiment analysis
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sent = SentimentIntensityAnalyzer()

os.chdir(r"C:\Users\User\Downloads")

data = pd.read_csv("TCS_feedback.csv")
data.head()
#for pros col
polarity = [round(sent.polarity_scores(i)['compound'], 2) for i in data['Pros']]
data['sentiment_score'] = polarity
data
polarity
data['sentiment_score'] = polarity
data['sentiment_score']
data.head()

#for cons col
polarity1 = [round(sent.polarity_scores(i)['compound'], 2) for i in data['Cons']]
data['sentiment_score1'] = polarity
data
polarity
data['sentiment_score1'] = polarity1
data['sentiment_score1']
data.head()
#for pros col
import seaborn as sns
sns.histplot(data["sentiment_score"],bins=20);

data['Label'] = pd.cut(x=data['sentiment_score'], bins=[-0.75, 0, 0.30, 1],
                     labels=['Negative', 'Neutral', 'Positive'])
data['Label'] 
data.head()

data.to_excel('sentimenttcs.xlsx',index=False)

#for cons col
sns.histplot(data["sentiment_score1"],bins=20);

data['Label'] = pd.cut(x=data['sentiment_score1'], bins=[-1, 0, 0.30, 1],
                     labels=['Negative', 'Neutral', 'Positive'])
data['Label'] 
data.head()

data.to_excel('sentimenttcs1.xlsx',index=False)

#for pros 
data=pd.read_excel('sentimenttcs.xlsx')
data
data.info()
data['Pros']  ## took only text column
data['Pros'].info()

#for cons
data1=pd.read_excel('sentimenttcs1.xlsx')
data1
data1.info()
data1['Cons']  ## took only text column
data1['Cons'].info()

#for pros
data['Pros'] = data['Pros'].astype(str) ## converting the data into string
data['Pros']
data.head()

#for cons
data1['Cons'] = data1['Cons'].astype(str) ## converting the data into string
data1['Cons']
data1.head()
#Lower Case
data["Pros_lower"] = data["Pros"].str.lower()    ## converting the text into lower case
data

data1["Cons_lower"] = data1["Cons"].str.lower()    ## converting the text into lower case
data1

#Remove Punctuation
PUNCT_TO_REMOVE = string.punctuation
PUNCT_TO_REMOVE

def remove_punctuation(text_lower):
    """custom function to remove the punctuation"""
    return text_lower.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

data["Pros_lower"] = data["Pros_lower"].apply(lambda text_lower: remove_punctuation(text_lower))
data["Pros_lower"]

data1["Cons_lower"] = data1["Cons_lower"].apply(lambda text_lower: remove_punctuation(text_lower))
data1["Cons_lower"]
#Removal of stopwords
from nltk.corpus import stopwords
", ".join(stopwords.words('english'))
STOPWORDS = set(stopwords.words('english'))
STOPWORDS 
def remove_stopwords(text_lower):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text_lower).split() if word not in STOPWORDS])


data["Pros_lower"] = data["Pros_lower"].apply(lambda text_lower: remove_stopwords(text_lower))
data["Pros_lower"]

data1["Cons_lower"] = data1["Cons_lower"].apply(lambda text_lower: remove_stopwords(text_lower))
data1["Cons_lower"]
data1.head()
#Removal of Frequent words
from collections import Counter
cnt = Counter()
for text_lower in data["Pros_lower"].values:
    for word in text_lower.split():
        cnt[word] += 1
cnt
cnt.most_common
cnt.most_common(10)
FREQWORDS = set([w for (w, wc) in cnt.most_common(10)])
FREQWORDS
def remove_freqwords(text_lower):
    """custom function to remove the frequent words"""
    return " ".join([word for word in str(text_lower).split() if word not in FREQWORDS])
data["Pros_lower"] = data["Pros_lower"].apply(lambda text_lower: remove_freqwords(text_lower))
data["Pros_lower"]

data1["Cons_lower"] = data1["Cons_lower"].apply(lambda text_lower: remove_freqwords(text_lower))
data1["Cons_lower"]
#Removal of Rare words
n_rare_words = 10
RAREWORDS = set([w for (w, wc) in cnt.most_common()[:-n_rare_words-1:-1]])
RAREWORDS
def remove_rarewords(text_lower):
    """custom function to remove the rare words"""
    return " ".join([word for word in str(text_lower).split() if word not in RAREWORDS])
data["Pros_lower"] = data["Pros_lower"].apply(lambda text_lower: remove_rarewords(text_lower))
data["Pros_lower"]

data1["Cons_lower"]= data1["Cons_lower"].apply(lambda text_lower: remove_rarewords(text_lower))
data1["Cons_lower"]
#Stemming
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
def stem_words(text_lower):
    return " ".join([stemmer.stem(word) for word in text_lower.split()])
data["Pros_lower"] = data["Pros_lower"].apply(lambda text_lower: stem_words(text_lower))
data["Pros_lower"] 

data1["Cons_lower"] = data1["Cons_lower"].apply(lambda text_lower: stem_words(text_lower))
data1["Cons_lower"] 

from nltk.stem.snowball import SnowballStemmer
SnowballStemmer.languages
#wordcloud
data
negative_df = data[data['Label'] == 'Negative']
negative_df
positive_df=data[data['Label']=='Positive']
positive_df
neutral_df=data[data['Label']=='Neutral']
neutral_df
#NEGATIVE LABEL WORDCLOUD
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
text_spam = ["".join(text) for text in negative_df['Pros_lower']]
final_text = "".join(text_spam)
#final_text[:500]
final_text
wordcloud_spam = WordCloud().generate(final_text)
plt.figure(figsize=(20,20))
plt.imshow(wordcloud_spam)
plt.axis("off")
plt.show()

#for cons_lower
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
text_spam = ["".join(text) for text in negative_df['Cons_lower']]
final_text = "".join(text_spam)
#final_text[:500]
final_text
wordcloud_spam = WordCloud().generate(final_text)
plt.figure(figsize=(20,20))
plt.imshow(wordcloud_spam)
plt.axis("off")
plt.show()

#POSITIVE LABELS WORDCLOUD
text_spam = ["".join(text) for text in positive_df['Pros_lower']]
final_text = "".join(text_spam)
#final_text[:500]
final_text
wordcloud_spam = WordCloud().generate(final_text)
plt.figure(figsize=(20,20))
plt.imshow(wordcloud_spam)
plt.axis("off")
plt.show()

#for conslower
text_spam = ["".join(text) for text in positive_df['Cons_lower']]
final_text = "".join(text_spam)
#final_text[:500]
final_text
wordcloud_spam = WordCloud().generate(final_text)
plt.figure(figsize=(20,20))
plt.imshow(wordcloud_spam)
plt.axis("off")
plt.show()
#NEUTRAL LABELS WORDCLOUD
text_spam = ["".join(text) for text in neutral_df['Pros_lower']]
final_text = "".join(text_spam)
#final_text[:500]
final_text
wordcloud_spam = WordCloud().generate(final_text)
plt.figure(figsize=(20,20))
plt.imshow(wordcloud_spam)
plt.axis("off")
plt.show()

#for conslower
text_spam = ["".join(text) for text in neutral_df['Cons_lower']]
final_text = "".join(text_spam)
#final_text[:500]
final_text
wordcloud_spam = WordCloud().generate(final_text)
plt.figure(figsize=(20,20))
plt.imshow(wordcloud_spam)
plt.axis("off")
plt.show()
#BIAGRAM FOR POSITIVE LABELS
positive_df['Pros_lower']
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
def get_ngrams(text, n=2):
    text = str(text)
    n_grams = ngrams(text.split(), n)
    returnVal = []
    try:
        for grams in n_grams:
            returnVal.append('_'.join(grams))
    except(RuntimeError):
        pass
    return ' '.join(returnVal).strip()

positive = positive_df["Pros_lower"].apply(get_ngrams, n=2)
positive
Bigram_string_list = positive.tolist()
bigram_string = ' '.join(Bigram_string_list)

Bigram_string_list
bigram_string
from wordcloud import WordCloud   # for the wordcloud
wordcloud = WordCloud(width = 2000, height = 1334, random_state=1,
                      background_color='black', colormap='Pastel1',
                      max_words = 75, collocations=False, normalize_plurals=False).generate(bigram_string)

# create the wordcloud
import matplotlib.pyplot as plt   # for wordclouds & charts
from matplotlib.pyplot import figure

# Define a function to plot word cloud
def plot_cloud(wordcloud):
    fig = plt.figure(figsize=(25, 17), dpi=80)
    plt.tight_layout(pad=0)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.box(False)
    plt.show()
    plt.close()

#Plot
plot_cloud(wordcloud)

#for conslower
positive_df['Cons_lower']
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
def get_ngrams(text, n=2):
    text = str(text)
    n_grams = ngrams(text.split(), n)
    returnVal = []
    try:
        for grams in n_grams:
            returnVal.append('_'.join(grams))
    except(RuntimeError):
        pass
    return ' '.join(returnVal).strip()

positive = positive_df["Cons_lower"].apply(get_ngrams, n=2)
positive
Bigram_string_list = positive.tolist()
bigram_string = ' '.join(Bigram_string_list)

Bigram_string_list
bigram_string
from wordcloud import WordCloud   # for the wordcloud
wordcloud = WordCloud(width = 2000, height = 1334, random_state=1,
                      background_color='black', colormap='Pastel1',
                      max_words = 75, collocations=False, normalize_plurals=False).generate(bigram_string)

# create the wordcloud
import matplotlib.pyplot as plt   # for wordclouds & charts
from matplotlib.pyplot import figure

# Define a function to plot word cloud
def plot_cloud(wordcloud):
    fig = plt.figure(figsize=(25, 17), dpi=80)
    plt.tight_layout(pad=0)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.box(False)
    plt.show()
    plt.close()

#Plot
plot_cloud(wordcloud)

#TRIGRAM OF POSITIVE LABELS
positive_df['Pros_lower']
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
def get_ngrams(text, n=3):
    text = str(text)
    n_grams = ngrams(text.split(), n)
    returnVal = []
    try:
        for grams in n_grams:
            returnVal.append('_'.join(grams))
    except(RuntimeError):
        pass
    return ' '.join(returnVal).strip()

tripositve= positive_df["Pros_lower"].apply(get_ngrams, n=3)
tripositve
Trigram_string_list =tripositve.tolist()
Trigram_string = ' '.join(Trigram_string_list)
Trigram_string_list
Trigram_string

from wordcloud import WordCloud   # for the wordcloud
wordcloud = WordCloud(width = 2000, height = 1334, random_state=1,
                      background_color='black', colormap='Pastel1',
                      max_words = 75, collocations=False, normalize_plurals=False).generate(Trigram_string )
# create the wordcloud
import matplotlib.pyplot as plt   # for wordclouds & charts
from matplotlib.pyplot import figure

# Define a function to plot word cloud
def plot_cloud(wordcloud):
    fig = plt.figure(figsize=(25, 17), dpi=80)
    plt.tight_layout(pad=0)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.box(False)
    plt.show()
    plt.close()

#Plot
plot_cloud(wordcloud)

#for conslwer
positive_df['Cons_lower']
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
def get_ngrams(text, n=2):
    text = str(text)
    n_grams = ngrams(text.split(), n)
    returnVal = []
    try:
        for grams in n_grams:
            returnVal.append('_'.join(grams))
    except(RuntimeError):
        pass
    return ' '.join(returnVal).strip()

positive = positive_df["Cons_lower"].apply(get_ngrams, n=2)
positive
Bigram_string_list = positive.tolist()
bigram_string = ' '.join(Bigram_string_list)

Bigram_string_list
bigram_string
from wordcloud import WordCloud   # for the wordcloud
wordcloud = WordCloud(width = 2000, height = 1334, random_state=1,
                      background_color='black', colormap='Pastel1',
                      max_words = 75, collocations=False, normalize_plurals=False).generate(bigram_string)

# create the wordcloud

# Define a function to plot word cloud
def plot_cloud(wordcloud):
    fig = plt.figure(figsize=(25, 17), dpi=80)
    plt.tight_layout(pad=0)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.box(False)
    plt.show()
    plt.close()

#Plot
plot_cloud(wordcloud)

#BIGRAM FOR NEGATIVE LABELS
negative_df['Pros_lower']
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
def get_ngrams(text, n=2):
    text = str(text)
    n_grams = ngrams(text.split(), n)
    returnVal = []
    try:
        for grams in n_grams:
            returnVal.append('_'.join(grams))
    except(RuntimeError):
        pass
    return ' '.join(returnVal).strip()
negative = negative_df["Pros_lower"].apply(get_ngrams, n=2)
negative
Bigram_string_list = negative.tolist()
bigram_string = ' '.join(Bigram_string_list)
Bigram_string_list
bigram_string
from wordcloud import WordCloud   # for the wordcloud
wordcloud = WordCloud(width = 2000, height = 1334, random_state=1,
                      background_color='black', colormap='Pastel1',
                      max_words = 75, collocations=False, normalize_plurals=False).generate(bigram_string)
# create the wordcloud
import matplotlib.pyplot as plt   # for wordclouds & charts
from matplotlib.pyplot import figure

# Define a function to plot word cloud
def plot_cloud(wordcloud):
    fig = plt.figure(figsize=(25, 17), dpi=80)
    plt.tight_layout(pad=0)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.box(False)
    plt.show()
    plt.close()

#Plot
plot_cloud(wordcloud)

#for cons
negative_df['Cons_lower']
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
def get_ngrams(text, n=2):
    text = str(text)
    n_grams = ngrams(text.split(), n)
    returnVal = []
    try:
        for grams in n_grams:
            returnVal.append('_'.join(grams))
    except(RuntimeError):
        pass
    return ' '.join(returnVal).strip()
negative = negative_df["Cons_lower"].apply(get_ngrams, n=2)
negative
Bigram_string_list = negative.tolist()
bigram_string = ' '.join(Bigram_string_list)
Bigram_string_list
bigram_string
from wordcloud import WordCloud   # for the wordcloud
wordcloud = WordCloud(width = 2000, height = 1334, random_state=1,
                      background_color='black', colormap='Pastel1',
                      max_words = 75, collocations=False, normalize_plurals=False).generate(bigram_string)
# create the wordcloud

# Define a function to plot word cloud
def plot_cloud(wordcloud):
    fig = plt.figure(figsize=(25, 17), dpi=80)
    plt.tight_layout(pad=0)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.box(False)
    plt.show()
    plt.close()

#Plot
plot_cloud(wordcloud)
#TRIGRAM FOR NEGATIVE LABELS
negative_df['Pros_lower']
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
def get_ngrams(text, n=3):
    text = str(text)
    n_grams = ngrams(text.split(), n)
    returnVal = []
    try:
        for grams in n_grams:
            returnVal.append('_'.join(grams))
    except(RuntimeError):
        pass
    return ' '.join(returnVal).strip()
trinegative= negative_df["Pros_lower"].apply(get_ngrams, n=3)
trinegative
Trigram_string_list =trinegative.tolist()
Trigram_string = ' '.join(Trigram_string_list)
Trigram_string

from wordcloud import WordCloud   # for the wordcloud
wordcloud = WordCloud(width = 2000, height = 1334, random_state=1,
                      background_color='black', colormap='Pastel1',
                      max_words = 75, collocations=False, normalize_plurals=False).generate(Trigram_string )
# create the wordcloud


# Define a function to plot word cloud
def plot_cloud(wordcloud):
    fig = plt.figure(figsize=(25, 17), dpi=80)
    plt.tight_layout(pad=0)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.box(False)
    plt.show()
    plt.close()

#Plot
plot_cloud(wordcloud)

#for cons
negative_df['Cons_lower']
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
def get_ngrams(text, n=3):
    text = str(text)
    n_grams = ngrams(text.split(), n)
    returnVal = []
    try:
        for grams in n_grams:
            returnVal.append('_'.join(grams))
    except(RuntimeError):
        pass
    return ' '.join(returnVal).strip()
trinegative= negative_df["Cons_lower"].apply(get_ngrams, n=3)
trinegative
Trigram_string_list =trinegative.tolist()
Trigram_string = ' '.join(Trigram_string_list)
Trigram_string

from wordcloud import WordCloud   # for the wordcloud
wordcloud = WordCloud(width = 2000, height = 1334, random_state=1,
                      background_color='black', colormap='Pastel1',
                      max_words = 75, collocations=False, normalize_plurals=False).generate(Trigram_string )
# create the wordcloud


# Define a function to plot word cloud
def plot_cloud(wordcloud):
    fig = plt.figure(figsize=(25, 17), dpi=80)
    plt.tight_layout(pad=0)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.box(False)
    plt.show()
    plt.close()

#Plot
plot_cloud(wordcloud)

#BIGRAM FOR NEUTRAL LABELS
neutral_df['Pros_lower']
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
def get_ngrams(text, n=2):
    text = str(text)
    n_grams = ngrams(text.split(), n)
    returnVal = []
    try:
        for grams in n_grams:
            returnVal.append('_'.join(grams))
    except(RuntimeError):
        pass
    return ' '.join(returnVal).strip()
neutral = neutral_df["Pros_lower"].apply(get_ngrams, n=2)
neutral
Bigram_string_list = neutral.tolist()
bigram_string = ' '.join(Bigram_string_list)
bigram_string

from wordcloud import WordCloud   # for the wordcloud
wordcloud = WordCloud(width = 2000, height = 1334, random_state=1,
                      background_color='black', colormap='Pastel1',
                      max_words = 75, collocations=False, normalize_plurals=False).generate(bigram_string)
# create the wordcloud


# Define a function to plot word cloud
def plot_cloud(wordcloud):
    fig = plt.figure(figsize=(25, 17), dpi=80)
    plt.tight_layout(pad=0)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.box(False)
    plt.show()
    plt.close()

#Plot
plot_cloud(wordcloud)

#for cons
neutral_df['Cons_lower']
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
def get_ngrams(text, n=2):
    text = str(text)
    n_grams = ngrams(text.split(), n)
    returnVal = []
    try:
        for grams in n_grams:
            returnVal.append('_'.join(grams))
    except(RuntimeError):
        pass
    return ' '.join(returnVal).strip()
neutral = neutral_df["Cons_lower"].apply(get_ngrams, n=2)
neutral
Bigram_string_list = neutral.tolist()
bigram_string = ' '.join(Bigram_string_list)
bigram_string

from wordcloud import WordCloud   # for the wordcloud
wordcloud = WordCloud(width = 2000, height = 1334, random_state=1,
                      background_color='black', colormap='Pastel1',
                      max_words = 75, collocations=False, normalize_plurals=False).generate(bigram_string)
# create the wordcloud


# Define a function to plot word cloud
def plot_cloud(wordcloud):
    fig = plt.figure(figsize=(25, 17), dpi=80)
    plt.tight_layout(pad=0)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.box(False)
    plt.show()
    plt.close()

#Plot
plot_cloud(wordcloud)


#TRIGRAM FOR NEUTRAL LABELS
neutral_df['Pros_lower']
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
def get_ngrams(text, n=3):
    text = str(text)
    n_grams = ngrams(text.split(), n)
    returnVal = []
    try:
        for grams in n_grams:
            returnVal.append('_'.join(grams))
    except(RuntimeError):
        pass
    return ' '.join(returnVal).strip()
trineutral= neutral_df["Pros_lower"].apply(get_ngrams, n=3)
trineutral
Trigram_string_list =trineutral.tolist()
Trigram_string = ' '.join(Trigram_string_list)
Trigram_string

from wordcloud import WordCloud   # for the wordcloud
wordcloud = WordCloud(width = 2000, height = 1334, random_state=1,
                      background_color='black', colormap='Pastel1',
                      max_words = 75, collocations=False, normalize_plurals=False).generate(Trigram_string )

# create the wordcloud


# Define a function to plot word cloud
def plot_cloud(wordcloud):
    fig = plt.figure(figsize=(25, 17), dpi=80)
    plt.tight_layout(pad=0)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.box(False)
    plt.show()
    plt.close()

#Plot
plot_cloud(wordcloud)

#for cons
neutral_df['Cons_lower']
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
def get_ngrams(text, n=3):
    text = str(text)
    n_grams = ngrams(text.split(), n)
    returnVal = []
    try:
        for grams in n_grams:
            returnVal.append('_'.join(grams))
    except(RuntimeError):
        pass
    return ' '.join(returnVal).strip()
trineutral= neutral_df["Cons_lower"].apply(get_ngrams, n=3)
trineutral
Trigram_string_list =trineutral.tolist()
Trigram_string = ' '.join(Trigram_string_list)
Trigram_string

from wordcloud import WordCloud   # for the wordcloud
wordcloud = WordCloud(width = 2000, height = 1334, random_state=1,
                      background_color='black', colormap='Pastel1',
                      max_words = 75, collocations=False, normalize_plurals=False).generate(Trigram_string )

# create the wordcloud


# Define a function to plot word cloud
def plot_cloud(wordcloud):
    fig = plt.figure(figsize=(25, 17), dpi=80)
    plt.tight_layout(pad=0)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.box(False)
    plt.show()
    plt.close()

#Plot
plot_cloud(wordcloud)