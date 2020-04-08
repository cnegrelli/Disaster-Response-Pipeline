import sys
import nltk
nltk.download(['punkt','wordnet'])
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.externals import joblib 


def load_data(database_filepath):

    '''
    This function loads the table from the SQL data base and
    separates the feature and the target variables. Also replaces some
    wrong values and records the names of the targets.

    input:
    database_filepath: SQL data base. Must include path if not in 
                       the same folder.

    output:
    X: Pandas DataFrame with the features for the ML model, in this 
       case the messages text.
    Y: Pandas DataFrame with the multiple targets.
    category_names: Name of the targets.
    '''

    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('MessagesLabeled', engine)
    df.related.replace(2,1,inplace=True)
    X = df['message'] 
    Y = df.drop(['id','message','original','genre'],axis=1)
    category_names = Y.columns.values
    return X, Y, category_names


def tokenize(text):
    '''
    This function case normalize, lemmatize and tokenize text.

    imput: 
    text: message.

    output: 
    clean_tokens: list of tokens case normalized and lemmatized.
    '''

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)
        
    return clean_tokens



def build_model():
    
    '''
    This function builds the ML model. Uses a pipeline that:
    a) vectorize the message making use of the tokenize function
    b) calculates the TF-IDF
    c) performs a multi-output classification.
    Then runs a Grid Search to find the best parameters for the model.

    input: None
    
    output: 
    cv: Best model
    '''

    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf',MultiOutputClassifier(RandomForestClassifier()))])

    parameters = {'clf__estimator__n_estimators': [100,150],
                  'clf__estimator__min_samples_split': [2,4]}

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, n_jobs=-1)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    
    '''
    This function evaluates the model and prints the
    overall accuracy and the classification report for each target.

    input:
    model: ML model.
    X_test: DataFrame with the feature variable.
    Y_test: DataFrane with the target variables.
    category_names: names of the target variables.

    output: None
    '''

    Y_pred = model.predict(X_test)
    print(model.best_params_)
    overall_accuracy = (Y_pred == Y_test).mean().mean()
    print(overall_accuracy)
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    '''
    This function saves the model to a pickle file.

    input:
    model: model to be saved.
    model_filepath: name of the pickle file. Must include path if not in folder.

    output: None
    '''

    joblib.dump(model, '{}'.format(model_filepath))


def main():
    '''
    This function reads the two filepaths needed and runs the other functions
    '''
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
