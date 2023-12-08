import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('spam_ham_dataset.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Vectorize the email text using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize classifiers
classifiers = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'SVM': SVC()
}

# Train and predict using each classifier
predictions = {}
for model, classifier in classifiers.items():
    classifier.fit(X_train_tfidf, y_train)
    predictions[model] = classifier.predict(X_test_tfidf)
    
# Evaluate the performance of each model
for model in classifiers:
    print(f"{model} Accuracy: {accuracy_score(y_test, predictions[model])}")
    print(f"{model} Classification Report:\n{classification_report(y_test, predictions[model])}")
    print(f"{model} Confusion Matrix:\n{confusion_matrix(y_test, predictions[model])}\n")

# Visualize Confusion Matrices
plt.figure(figsize=(12, 10))
for i, model in enumerate(classifiers, 1):
    plt.subplot(2, 2, i)
    cm = confusion_matrix(y_test, predictions[model])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
    plt.title(f'{model} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
plt.tight_layout()
plt.show()

# Model Comparison Bar Plot
accuracies = [accuracy_score(y_test, predictions[model]) for model in classifiers]

plt.figure(figsize=(10, 6))
plt.bar(classifiers.keys(), accuracies, color=['blue', 'orange', 'green', 'red'])
plt.ylim([0, 1])
plt.ylabel('Accuracy')
plt.title('Model Comparison Tf-Idf')
plt.show()

#Preprocesing and training model
import pandas as pd
# Load the dataset
data = pd.read_csv('spam_ham_dataset.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Vectorize the email text using CountVectorizer
count_vectorizer = CountVectorizer()
X_train_count = count_vectorizer.fit_transform(X_train)
X_test_count = count_vectorizer.transform(X_test)

# Initialize classifiers
classifiers = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),  # Increase max_iter
    'Decision Tree': DecisionTreeClassifier(),
    'SVM': SVC()
}

# Train and predict using each classifier
predictions = {}
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    for model, classifier in classifiers.items():
        classifier.fit(X_train_count, y_train)
        predictions[model] = classifier.predict(X_test_count)

        
#Preprocesing and training model

import pandas as pd
# Load the dataset
data = pd.read_csv('spam_ham_dataset.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Vectorize the email text using CountVectorizer
count_vectorizer = CountVectorizer()
X_train_count = count_vectorizer.fit_transform(X_train)
X_test_count = count_vectorizer.transform(X_test)

# Initialize classifiers
classifiers = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),  # Increase max_iter
    'Decision Tree': DecisionTreeClassifier(),
    'SVM': SVC()
}

# Train and predict using each classifier
predictions = {}
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    for model, classifier in classifiers.items():
        classifier.fit(X_train_count, y_train)
        predictions[model] = classifier.predict(X_test_count)