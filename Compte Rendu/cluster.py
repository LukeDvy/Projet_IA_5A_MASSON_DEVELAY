import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize

# Load the data
data = pd.read_csv('archive/datas.csv')

# Convert 'Book-Title' column to string type (if not already)
data['Book-Title'] = data['Book-Title'].astype(str)

# Tokenize and perform part-of-speech tagging
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def get_nouns(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    nouns = [word for word, pos in pos_tags if pos.startswith('N')]
    return ' '.join(nouns)

# Apply part-of-speech tagging to 'Book-Title'
data['Nouns'] = data['Book-Title'].apply(get_nouns)

# Vectorize the text (nouns) using CountVectorizer
vectorizer = CountVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(data['Nouns'])

# Apply K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

# Map numerical cluster labels to custom tags
cluster_tags = {0: 'Adventure', 1: 'Romance', 2: 'Mystery', 3: 'Science Fiction', 4: 'History'}
data['Cluster-Tag'] = data['Cluster'].map(cluster_tags)
# Save the modified dataset to the original CSV file
data.to_csv('archive/final_datas.csv', index=False)
