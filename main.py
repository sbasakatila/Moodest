import re
import pickle
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from gensim.parsing.preprocessing import remove_stopwords

# Dataset = id,tweet,label
dataset = pd.read_csv(r'C:\Users\W10USER\PycharmProjects\MoodestSentiment\Corona_NLP_train.csv',
                      sep="," ,encoding="ISO-8859-1")

# Removing the unnecessary columns
dataset = dataset[['OriginalTweet','Sentiment']]

# Storing data in lists
tweet, label = list(dataset['OriginalTweet']), list(dataset['Sentiment'])

# Defining emojis
emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad',
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed',
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink',
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}


def preprocess(textdata):
    processedText = []
    wordLemm = WordNetLemmatizer()

    # Defining regex patterns.
    urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern = '@[^\s]+'
    alphaPattern = "[^a-zA-Z0-9]"
    sequencePattern = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"

    for tweet in textdata:
        # String
        tweet = str(tweet)
        # Lowers letters
        tweet = tweet.lower()
        # Replace all URls with ' '
        tweet = re.sub(urlPattern, ' ', tweet)
        # Replace all emojis
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, emojis[emoji])
        # Replace @USERNAME to ' '.
        tweet = re.sub(userPattern, ' ', tweet)
        # Replace all non alphabets
        tweet = re.sub(alphaPattern, " ", tweet)
        # Replace repeating letters by 2 letters
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)
        # Checking stopwords and Lemmatizing
        tweetwords = ''
        for word in tweet.split():
            if len(word) > 1:
                word = wordLemm.lemmatize(word)
                tweetwords += (word + ' ')

        tweet = remove_stopwords(tweetwords)

        processedText.append(tweet)

    return processedText


processedText = preprocess(tweet)

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(processedText, label,
                                                    test_size = 0.2, random_state = 0)

# Data transformation
vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectoriser.fit(X_train)

X_train = vectoriser.transform(X_train)
X_test  = vectoriser.transform(X_test)

# Modelling
def model_Evaluate(model):
    # Predict values for Test dataset
    y_pred = model.predict(X_test)

    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)

    categories = ['Extremely Negative','Negative','Neutral','Positive','Extremely Positive']
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='',
                xticklabels=categories, yticklabels=categories)

    plt.xlabel("Predicted values", fontdict={'size': 14}, labelpad=10)
    plt.ylabel("Actual values", fontdict={'size': 14}, labelpad=10)
    plt.title("Confusion Matrix", fontdict={'size': 18}, pad=20)


LRmodel = LogisticRegression(C = 5, max_iter = 1000, n_jobs=-1)
LRmodel.fit(X_train, y_train)

model_Evaluate(LRmodel)

file = open('vectoriser-ngram-(1,2).pickle','wb')
pickle.dump(vectoriser, file)
file.close()


def load_models():

    # Load the vectoriser.
    file = open(r'C:\Users\W10USER\PycharmProjects\Moodest2\vectoriser-ngram-(1,2).pickle', 'rb')
    vectoriser = pickle.load(file)
    file.close()
    # Load the LR Model.
    file = open(r'C:\Users\W10USER\PycharmProjects\Moodest2\Sentiment-LR.pickle', 'rb')
    LRmodel = pickle.load(file)
    file.close()

    return vectoriser, LRmodel


def predict(vectoriser, model, tweet):
    # Predict the sentiment
    textdata = vectoriser.transform(preprocess(tweet))
    label = model.predict(textdata)

    # Make a list of text with sentiment.
    data = []
    for tweet, pred in zip(tweet, label):
        data.append((tweet, pred))

    # Convert the list into a Pandas DataFrame.
    df = pd.DataFrame(data, columns=['tweet', 'label'])
    df = df.replace([-1,0,1,2,3], ['Extremely Negative','Negative','Neutral','Positive','Extremely Positive'])
    return df

if __name__ == "__main__":

    platform=input('''       Select a platform
            1) TWITTER
            2) REDDIT
            3) Sentence''')

    if platform=='1':
        dataset = pd.read_csv(r'C:\Users\W10USER\PycharmProjects\MoodestSentiment\Workflow 154325 - twitter-698589.csv',
                              sep=',')

        dataset = dataset[['text']]
        text = list(dataset['text'])

        word=input("Enter a Hashtag or word")
        preprocess(text)
        processedText = preprocess(text)
        processedText = [s for s in processedText if word in s]


        df = predict(vectoriser, LRmodel, processedText)

        plt.figure(figsize=(10, 5))
        sns.countplot(x='label', data=df,
                            order=['Extremely Negative', 'Negative', 'Neutral', 'Positive', 'Extremely Positive'])
        plt.title("Sentiment")
        plt.ylabel("Count", fontsize=12)
        plt.xlabel("Sentiments", fontsize=12)
        plt.show()

    elif platform=='2':

        input("Enter a Hashtag or word")
        dataset = pd.read_csv(r'C:\Users\W10USER\PycharmProjects\MoodestSentiment\coronavirus_reddit_clean_comments.csv',
                              sep=',')

        dataset = dataset[['comment']]
        text = list(dataset['comment'])

        word=input("Enter a Hashtag or word")
        preprocess(text)
        processedText = preprocess(text)
        processedText = [s for s in processedText if word in s]


        df = predict(vectoriser, LRmodel, processedText)

        plt.figure(figsize=(10, 5))
        sns.countplot(x='label', data=df,
                      order=['Extremely Negative', 'Negative', 'Neutral', 'Positive', 'Extremely Positive'])
        plt.title("Sentiment")
        plt.ylabel("Count", fontsize=12)
        plt.xlabel("Sentiments", fontsize=12)
        plt.show()

    elif platform=='3':
        sentence=[]
        text=input("Please enter a sentence:")
        sentence.append(text)
        preprocess(sentence)
        processedText = preprocess(sentence)

        df = predict(vectoriser, LRmodel, processedText)
        print(df.head())
