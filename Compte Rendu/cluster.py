import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize

# Charger les données
data = pd.read_csv('archive/datas.csv')
data['Book-Title'] = data['Book-Title'].astype(str)

# Tokeniser et effectuer l'étiquetage de parties du discours (POS tagging)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def get_nouns(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    nouns = [word for word, pos in pos_tags if pos.startswith('N')]
    return ' '.join(nouns)

# Appliquer l'étiquetage de parties du discours (POS tagging) à la colonne 'Book-Title'
data['Nouns'] = data['Book-Title'].apply(get_nouns)

# Vectoriser le texte (noms) en utilisant CountVectorizer
vectorizer = CountVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(data['Nouns'])

# Appliquer le clustering K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

# Mapper les étiquettes numériques de cluster à des tags personnalisés
cluster_tags = {0: 'Aventure', 1: 'Romance', 2: 'Mystère', 3: 'Science Fiction', 4: 'Histoire'}
data['Cluster-Tag'] = data['Cluster'].map(cluster_tags)

# Sauvegarder le jeu de données modifié dans le fichier CSV d'origine
data.to_csv('archive/final_datas.csv', index=False)
