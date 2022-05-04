import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from joblib import dump

from nlp.nlp_module import clean_message, fit_vectorize_messages


#Load data
df = pd.read_csv('../datasets/FinalBalancedDataset.csv', names = [
    'id',
    'target',
    'tweet'
])[1:]

df['clean_tweet'] = df['tweet'].apply(clean_message)

#Train/Test Split
x = df['clean_tweet'].values
y = df['target'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=32)

X_train = fit_vectorize_messages(x_train)
X_test = fit_vectorize_messages(x_test)

classifier = LogisticRegression(solver='sag', max_iter=1000000000)
classifier.fit(X_train, y_train)

score = classifier.score(X_test, y_test)
print("L2-sag")
print("Accuracy:", score)

dump(classifier, '../models/binary_classifier.joblib') 