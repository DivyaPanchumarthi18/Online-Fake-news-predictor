#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


import seaborn as sns


# In[5]:


from sklearn.feature_extraction.text import CountVectorizer


# In[6]:


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# In[9]:


#read datasets
fake = pd.read_csv("C:/Users/HP/Downloads/Fake.csv")
true = pd.read_csv("C:/Users/HP/Downloads/True.csv")


# In[10]:


fake.shape


# In[11]:


true.shape


# In[12]:


fake['target'] = 'fake'
true['target'] = 'true'


# In[13]:


fake.head()


# In[15]:


true.head()


# In[16]:


#conctenate two datasets
data = pd.concat([fake,true]).reset_index(drop=True)
data.shape


# In[17]:


#we have to shuffle the data because all the fake data is in top and all true data is at bottom
#shuffling data
from sklearn.utils import shuffle
data = shuffle(data)
data=data.reset_index(drop=True)


# In[18]:


#checking if the data is shuffeled or not
data.head()


# In[19]:


data.info()


# In[20]:


#in this data the date is not needed so were removing the date
data.drop(["date"],axis=1,inplace=True)
data.head()


# In[21]:


#we dont need the title because we can get the title by reading the text so , were removing the title
#removing title
data.drop(["title"],axis=1,inplace=True)
data.head()


# In[22]:


# converting uppercase letters to lowercase letters
data['text']=data['text'].apply(lambda x: x.lower())
data.head()


# In[23]:


#remove punctuation (the words got repeated in a same sentence)
import string
def punctuation_removal(text):
    all_list=[char for char in text if char not in string.punctuation]
    clean_str=''.join(all_list)
    return clean_str

data['text'] = data['text'].apply(punctuation_removal)


# In[24]:


#check
data.head()


# In[25]:


#removing stopwords
import nltk   #nltk is natural language tool kit
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


# In[26]:


data.head()


# In[27]:


# How many articles per subject?
print(data.groupby(['subject'])['text'].count())
data.groupby(['subject'])['text'].count().plot(kind="bar")
plt.show()


# In[28]:


# How many fake and real articles?
print(data.groupby(['target'])['text'].count())
data.groupby(['target'])['text'].count().plot(kind="bar")
plt.show()


# In[29]:


get_ipython().system('pip install wordcloud')


# In[30]:


# Word cloud for fake news (shows a graphical image of he most used words in fake news)
from wordcloud import WordCloud

fake_data = data[data["target"] == "fake"]
all_words = ' '.join([text for text in fake_data.text])

wordcloud = WordCloud(width= 800, height= 500,
                          max_font_size = 110,
                          collocations = False).generate(all_words)

plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[31]:


# Word cloud for real news(shows a graphical image of he most used words in real news)
from wordcloud import WordCloud
real_data = data[data["target"] == "true"]
all_words = ' '.join([text for text in fake_data.text])
wordcloud = WordCloud(width= 800, height= 500,
                          max_font_size = 110,
                          collocations = False).generate(all_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[32]:


# Most frequent words counter   (used to find how many times the word is usd in the text)
from nltk import tokenize
token_space = tokenize.WhitespaceTokenizer()
def counter(text, column_text, quantity):
    all_words = ' '.join([text for text in text[column_text]])
    token_phrase = token_space.tokenize(all_words)
    frequency = nltk.FreqDist(token_phrase)
    df_frequency = pd.DataFrame({"Word": list(frequency.keys()),
                                   "Frequency": list(frequency.values())})
    df_frequency = df_frequency.nlargest(columns = "Frequency", n = quantity)
    plt.figure(figsize=(12,8))
    ax = sns.barplot(data = df_frequency, x = "Word", y = "Frequency", color = 'blue')
    ax.set(ylabel = "Count")
    plt.xticks(rotation='vertical')
    plt.show()


# In[33]:


# Most frequent words in fake news
counter(data[data["target"] == "fake"], "text", 20)


# In[34]:


# Most frequent words in real news
counter(data[data["target"] == "true"], "text", 20)


# In[35]:


# Function to plot the confusion matrix
from sklearn import metrics
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[36]:


#spliting the data (20% as testing and 80%as training) 
X_train,X_test,y_train,y_test = train_test_split(data['text'], data.target, test_size=0.2, random_state=42)


# In[37]:


X_train.head()


# In[38]:


y_train.head()


# In[39]:


#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
# Vectorizing and applying TF-IDF
pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', DecisionTreeClassifier(criterion= 'entropy',
                                           max_depth = 20, 
                                           splitter='best', 
                                           random_state=42))])


# In[40]:


# Fitting the model
model = pipe.fit(X_train, y_train)


# In[41]:


# Accuracy
prediction = model.predict(X_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))


# In[42]:


cm = metrics.confusion_matrix(y_test, prediction)
plot_confusion_matrix(cm, classes=['Fake', 'Real'])


# In[ ]:




