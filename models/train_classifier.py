# import libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import re

import sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin

import pickle

def load_data(database_filepath):
    """
    Load Data from the Database Function
    
    Arguments:
        database_filepath -> Path to SQLite destination database
    Output:
        X -> dataframe containing features
        Y -> dataframe containing labels
        category_names -> List of categories name
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('message', engine)
    df['related'] = df['related'].map(lambda x: 1 if x == 2 else x)
    df = df.drop('child_alone', axis=1)
    X = df['message']
    y = df.iloc[:, 4:]
    category_names = y.columns
    return X, y, category_names

def tokenize(text):
    """
    tokenize the text Function
    
    Arguments:
        text -> text message to be tokenized
    Output:
        clean_tokens -> list of clean tokens
    """
    # normalize text
    text = text.lower()
    stop_words = stopwords.words("english")
    
    # tokenize
    words = word_tokenize(text)
    
    # stemming
    stemmed = [PorterStemmer().stem(w) for w in words]
    
    # lemmatizing
    clean_tokens = [WordNetLemmatizer().lemmatize(w) for w in stemmed if w not in stop_words]
    
    return clean_tokens


def build_model():
    """
    Build pipeline function
    
    Output:
        a Scikit machine learning pipepline
    
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(min_samples_split=20)))
    ])
    
    return pipeline

def evaluate_model(model, X_test, y_test, category_names):
    """
    Evaluate pipeline function
    
    Arguments:
        model -> a scikit machine learning pipeline
        X_test -> test features
        y_test -> test labels
        category_names -> labels
    Output:
        classification report
    """
    y_pred_test = model.predict(X_test)
    # print(classification_report(y_true = y_test.values, y_pred = y_pred_test, target_names=category_names))
    
def save_model(model, model_filepath):
    """
    Save pipeline function as pickle file
    
    Arguments:
        model -> Scikit pipeline
        model_filepath -> destination path 
    """
    model_pkl = open(model_filepath, 'wb')
    pickle.dump(model, model_pkl)
    model_pkl.close()

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()