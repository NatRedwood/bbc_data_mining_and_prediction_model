import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

import seaborn as sns
import os
os.chdir("C:\\Users\\natal\\Python files")

data = pd.read_csv('bbc-text.csv')

data.head()

data['category_id'] = data['category'].factorize()[0] #To create a column with id for each category of texts

data.head()

data.columns

data.groupby('category').count().plot.bar(ylim=0)

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
stopwords

data['news_without_stopwords'] = data['text'].apply(lambda x:' '.join([word for word in x.split() if word not in (stopwords)]))
data.head()
data.shape

ps = PorterStemmer()
data['news_porter_stemmed'] = data['news_without_stopwords'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split()]))
data.head()
data['news_porter_stemmed'] = data['news_porter_stemmed'].apply(lambda x: ' '.join(x.lower() for x in x.split()))
data.head()
#data.drop(columns=['news_poter_stemmed']) #To drop a column

data['news_porter_stemmed'] = data['news_porter_stemmed'].str.replace('[^\w\s]','')
data.head()

freq = pd.Series(' '.join(data['news_porter_stemmed']).split()).value_counts()
freq.head()

freq2 = freq[freq <=3]
freq2.head()

freq3 = list(freq2.index.values)
freq3
freq3[:10]

data['news_porter_stemmed'] = data['news_porter_stemmed'].apply(lambda x: ' '.join([word for word in x.split() if word not in (freq3)]))
data.head()

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf = True, min_df=5, norm='l2', encoding = 'latin-1', ngram_range=(1,2))
dir(tfidf)
features = tfidf.fit_transform(data.news_porter_stemmed).toarray()

data.news_porter_stemmed.shape
features.shape
type(features)
features[0]
len(features[0])

labels=data.category_id
labels.shape

category_id_df = data[['category','category_id']].drop_duplicates().sort_values('category_id')
category_id_df

category_to_id = dict(category_id_df.values)
category_to_id

id_to_category = dict(category_id_df[['category_id','category']].values)
id_to_category

from sklearn.feature_selection import chi2
N=3
for category,category_id in sorted(category_to_id.items()):
    features_chi2 = chi2(features,labels == category_id) 
    indices = np.argsort(features_chi2[0])
    features_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in features_names if len(v.split(' ')) == 1]
    bigrams = [v for v in features_names if (v.split(' ')) == 2]
    print("#  '{}':".format(category))
    print("   .  Most correlated unigrams:\n       . {}".format('\n          .  '.join(unigrams[-N:])))
    print("   .  Most correlated bigrams:\n       . {}".format('\n          .  '.join(bigrams[-N:])))

from sklearn.manifold import TSNE 

SAMPLE_SIZE = int(len(features)*.3)
np.random.seed(0)
indices = np.random.choice(range(len(features)), size = SAMPLE_SIZE, replace = False)
projected_features = TSNE(n_components = 2, random_state=0).fit_transform(features[indices])
colors=['pink','green','midnightblue','orange','darkgrey']

for category, category_id in sorted(category_to_id.items()):
    points = projected_features[(labels[indices]==category_id).values]
    plt.scatter(points[:,0],points[:,1], s= 30, c=colors[category_id],label=category)
plt.title("tf-idf feature vector for each article, projected on 2 dimensions.", fontdict=dict(fontsize=15))
plt.legend()

points.shape

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

models = [ RandomForestClassifier(n_estimators = 200, max_depth = 3 , random_state = 0 ), MultinomialNB(), LogisticRegression(random_state = 0)]

data.head()
data.shape

CV = 5
cv_df = pd.DataFrame(index = range(CV*len(models)))
entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
sns.boxplot(x='model_name', y= 'accuracy', data=cv_df)
sns.stripplot(x='model_name', y= 'accuracy', data = cv_df , size = 8, jitter = True , edgecolor='gray', linewidth = 2)

features.shape

from sklearn.model_selection import train_test_split
model = LogisticRegression(random_state=0)
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, data.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)
y_pred

from sklearn.metrics import confusion_matrix
import seaborn as sns
conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.category.values, yticklabels=category_id_df.category.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')

model.fit(features, labels)

from sklearn.feature_selection import chi2
N = 5
for category, category_id in sorted(category_to_id.items()):
    indices = np.argsort(model.coef_[category_id])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
    bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
    print("# '{}':".format(category))
    print("  . Top unigrams:\n       . {}".format('\n       . '.join(unigrams)))
    print("  . Top bigrams:\n       . {}".format('\n       . '.join(bigrams)))
texts = ["Hooli stock price soared after a dip in PiedPiper revenue growth.",
         "Captain Tsubasa scores a magnificent goal for the Japanese team.",
         "Merryweather mercenaries are sent on another mission, as government oversight groups call for new sanctions.",
         "Beyoncé releases a new album, tops the charts in all of south-east Asia!",
         "You won't guess what the latest trend in data analysis is!"]
text_features = tfidf.transform(texts)

predictions = model.predict(text_features)
for text, predicted in zip(texts, predictions):
    print('"{}"'.format(text))
    print("  - Predicted as: '{}'".format(id_to_category[predicted]))
    print("")


