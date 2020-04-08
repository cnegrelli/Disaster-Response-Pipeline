import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):

    '''
    This function creates the DataFrame that will be used.

    input:
    messages_filepath: name of the csv file with the messeges 
                       to be classified. Must include the path
                       if not in the same folder.
    categories_filepath: name of the cvs files with the labels
                         for each message. Must include the path 
                         if not in the same folder.

    output:
    df: Pandas DataFrame with messages and categories merged.
    '''

    messages = pd.read_csv('{}'.format(messages_filepath))
    categories = pd.read_csv('{}'.format(categories_filepath))
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):

    '''
    This function cleans the DataFrame: 
    a) splits categories into separate columns
    b) renames de columns and assign the correct value (0 or 1)
    c) creates a DataFrame with the messages and this new columns
    d) drops duplicates.

    input:
    df: Pandas DataFrame.
    
    output:
    df: cleaned Pandas DataFrame.
    '''

    categories = df.categories.str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1:].astype(int)
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df,categories],axis=1)
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    
    '''
    This function saves de cleaned DataFrame to a SQL database.

    input:
    df: cleaned Pandas DataFrame.
    database_filenameL: name of the SQL database where the DataFrame
                        is going to be saved. Must include the path 
                        if not in folder.

    output: None
    '''

    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('MessagesLabeled', engine, index=False)
    pass  


def main():

    '''
    This function reads the three filepaths needed and runs the other functions
    '''

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
