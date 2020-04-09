# Disaster Response Pipeline Project

This project is part of the Udacity's Data Scientist Nanodegree.

### Overview

The goal of this project is to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages. 


### Instructions

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Files

- data/process_data.py: python script that reads two csv files (the messages file and the categories files) and creates a SQL
                 database with a cleaned table.

- data/disaster_messages.csv: csv file with the messages.

- data/disaster_categories.csv: csv table with the categories for each message.

- data/DisasterResponse.db: output of the process_data.py, you don't need this file to start.

- models/train_classifier.py: python script that reads the SQL database and creates and trains a classifier, and stores it in
                     a pickle file.
                     
- app/run.py: python scripts that runs the app.

- app/templates/go.html:

- app/templates/master.html:

