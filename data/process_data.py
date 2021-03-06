# import libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load message and categories files
    
    Arguments:
        messages_filepath -> path to the CSV file containing messages
        categories_filepath -> path to the CSV file containing categories
    
    Output:
        df -> dataframe containing messages and categories
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df
    
def clean_data(df):
    """
    Clean categories Data 
    
    Arguments:
        df -> dataframe containing messages and categories
    Output:
        df -> cleaned dataframe containing messages and categories
    """
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.loc[0]

    # use this row to extract a list of new column names for categories.
    row = row.str.split('-')
    category_colnames = row.apply(lambda x: x[0])

    # rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1:]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int) 

    # Replace `categories` column in `df` with new category columns. 
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], join='inner', axis=1)

    # convert 2 to 1
    categories[column] = categories[column].apply(lambda x: 1 if x == 2 else x)
    
    # Remove duplicates.
    df.drop_duplicates(inplace = True)
  
    return df

def save_data(df, database_filename):
    """
    Save Data to SQLite database 
    
    Arguments:
        df -> cleaned dataframe containing messages and categories
    Output:
        database_filename -> Path to SQLite database
    """

    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('message', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()