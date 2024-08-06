Name:- Dev shivhare
company:- Codtech IT Solution
ID:- CT6A1734
Domain:- Artificial Intelligence
Duration:- july-August
Mentor:- Neela santosh kumar

OVERVIEW OF TASK 1:
Project Overview: Fake News Detection Using NLP
This project aims to develop a machine learning model to detect fake news using Natural Language Processing (NLP) techniques. Below is an outline of the project's major steps, including data preprocessing, feature selection, model building, and evaluation.

Data Preprocessing
Import Libraries:

The project begins by importing necessary libraries such as pandas for data manipulation and nltk for natural language processing.
Load Dataset:

The dataset, news_articles.csv, is loaded into a DataFrame using pandas.
Data Inspection:

Basic data inspection is performed using head(), describe(), and info() methods to understand the dataset's structure and summary statistics.
Null values in the dataset are identified and subsequently filled with empty strings.
Feature Selection:

Columns deemed unnecessary for the analysis, such as author, published, main_img_url, type, hasImage, and site_url, are dropped from the DataFrame.
Text Processing
Install and Import NLTK:

nltk library is installed and imported. Necessary modules such as stopwords and PorterStemmer are also imported.
Stopwords are common words that are usually removed from the text as they do not contribute significantly to the model's prediction.
Text Cleaning and Stemming:

A function named stemming is defined to preprocess text by removing non-alphabetic characters, converting text to lowercase, splitting into words, removing stopwords, and applying stemming using PorterStemmer.
The stemming function is applied to the text data.
Model Building
Feature Extraction:

The text data (x) and corresponding labels (y) are separated.
The data is split into training and testing sets using train_test_split from sklearn.model_selection.
Vectorization:

TfidfVectorizer is used to convert text data into numerical features suitable for machine learning algorithms.
Training the Model:

A PassiveAggressiveClassifier from sklearn.linear_model is chosen as the machine learning model.
The model is trained using the training data.
Model Evaluation
Accuracy Calculation:
The trained model's accuracy is evaluated on the test data to measure its performance.
A confusion matrix is generated to further analyze the model's performance in classifying fake news correctly.
Conclusion
The project demonstrates the application of NLP techniques for preprocessing text data and the use of a machine learning model to detect fake news. The process includes data cleaning, feature extraction, model training, and evaluation, showcasing the end-to-end workflow of building a fake news detection system.
