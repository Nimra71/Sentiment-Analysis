📊 Restaurant Review Sentiment Analysis
Project Overview

This project performs sentiment analysis on restaurant reviews using Natural Language Processing (NLP) and Machine Learning.
It classifies reviews as positive, negative, or optionally neutral, based on text content.

The project uses two main scripts:

1: sentiment_analysis.py → Train, evaluate, and save models

2: sentiment_prediction.py → Predict sentiment on new/fresh reviews

Project Components

Data Files:

* a1_RestaurantReviews_HistoricDump.tsv → Training dataset

* a2_RestaurantReviews_FreshDump.tsv → New/fresh dataset for prediction

* c3_Predicted_Sentiments_Fresh_Dump.tsv → Output dataset with predicted labels

Model Files:

* c1_BoW_sentiment_Model.pkl → Bag of Words (CountVectorizer) model

* c2_classifier_sentiment_Model → Trained classifier

Python Scripts:

* sentiment_analysis.py → Preprocessing, BoW, train classifier, predict, compute accuracy & confusion matrix, save models

* sentiment_prediction.py → Load saved models, predict sentiment on fresh dataset, save results

  Workflow:

1: Data Preprocessing

Load dataset with pandas (.tsv, tab-separated)

Remove punctuation, lowercase, tokenize, remove stopwords, and stem words using NLTK

Store cleaned reviews in a corpus

2: Feature Extraction

Convert text to numeric features using Bag of Words (CountVectorizer)

Limit features to a fixed number (e.g., 1500)

3: Model Training & Evaluation

Train a classifier (e.g., Naive Bayes)

Predict sentiment on training data

Evaluate using accuracy and confusion matrix

Save:

BoW model → c1_BoW_sentiment_Model.pkl

Classifier → c2_classifier_sentiment_Model

Predicted training dataset → c3_Predicted_Sentiments_Fresh_Dump.tsv

4: Prediction on Fresh Data

Load a2_RestaurantReviews_FreshDump.tsv

Load saved models (BoW + classifier)

Preprocess fresh reviews

Transform text using saved BoW

Predict sentiment:

1 = Positive

0 = Negative

Optional Neutral

Save predictions to a new TSV file

Segment analysis/
├── a1_RestaurantReviews_HistoricDump.tsv
├── a2_RestaurantReviews_FreshDump.tsv
├── c1_BoW_sentiment_Model.pkl
├── c2_classifier_sentiment_Model
├── c3_Predicted_Sentiments_Fresh_Dump.tsv
├── sentiment_analysis.py
├── sentiment_prediction.py
├── README.md

How to Run

1 Install dependencies
pip install pandas numpy nltk scikit-learn joblib

2 Train & Evaluate
python sentiment_analysis.py
Outputs: trained models, predicted dataset, accuracy & confusion matrix

3 Predict on Fresh Data
python sentiment_prediction.py
Outputs: new TSV file with predicted sentiment labels

 Author
Nimra
