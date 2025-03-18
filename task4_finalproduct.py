#CODTECH IT SOLUTIONS. 
#TASK-4 MACHINE LEARNING MODEL IMPLEMENTATION.
#PROGRAM TO DETECT THE SPAM MAIL. 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
a = pd.read_csv('task4_support.csv')
b, c, d, e = train_test_split(a['email'], a['label'], test_size=0.2, random_state=42)
f = TfidfVectorizer()
g = f.fit_transform(b)
h = f.transform(c)
i = MultinomialNB()
i.fit(g, d)
j = i.predict(h)
k = accuracy_score(e, j)
print(f'Accuracy: {k:.3f}')
print('Classification Report:')
print(classification_report(e, j))
print('Confusion Matrix:')
print(confusion_matrix(e,j))
