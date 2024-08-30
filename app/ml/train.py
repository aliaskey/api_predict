import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Charger le dataset
data = pd.read_csv('app/data/training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None)
data.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']

# Prétraitement des données
data['target'] = data['target'].apply(lambda x: 1 if x == 4 else 0)

# Vectorisation du texte
vectorizer = CountVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['text'])
y = data['target']

# Séparation en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Sauvegarder le modèle et le vectorizer
joblib.dump(model, 'app/data/logistic_model.pkl')
joblib.dump(vectorizer, 'app/data/vectorizer.pkl')
