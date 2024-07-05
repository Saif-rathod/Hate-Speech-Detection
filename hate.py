import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import re
import nltk

from nltk.util import pr
stemmer=nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword=set(stopwords.words("english"))

data=pd.read_csv(r"C:\Users\Saif\ML Projects\Hate Speech Detection\twitter_data.csv")

data['labels']=data['class'].map({0:"Hate speech",1:"Not offensive",2:"Neutral"})

data = data[["tweet","labels"]]


def tokenize_stem(text):
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stopword]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [stemmer.stem(word) for word in tokens]
    return tokens


vectorizer = CountVectorizer(tokenizer=tokenize_stem)
X = vectorizer.fit_transform(data['tweet'])


X_train, X_test, y_train, y_test = train_test_split(X, data['labels'], test_size=0.2, random_state=42)

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

tweet = input("Enter your tweet: ")

tweet_vector = vectorizer.transform([tweet])
prediction = classifier.predict(tweet_vector)

print(f"The tweet is classified as: {prediction[0]}")

while True:
    response = input("Do you want to continue? (y/n): ").strip().lower()

    if response == 'y':
        tweet = input("Here uhh go!:")
        tweet_vector = vectorizer.transform([tweet])
        prediction = classifier.predict(tweet_vector)
        print(f"The tweet is classified as: {prediction[0]}")
    
    elif response == 'n':
        print("You chose no! Glad to see you... Visit again! :)")
        break
    else:
        print("Invalid response. Please enter 'y' for yes or 'n' for no.")

