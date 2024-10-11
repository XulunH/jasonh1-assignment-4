from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)


# TODO: Fetch dataset, initialize vectorizer and LSA here
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

stop_words = stopwords.words('english')

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

n_components = 100 
lsa = TruncatedSVD(n_components=n_components, random_state=0)
X_reduced = lsa.fit_transform(X)

def search_engine(query):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    # TODO: Implement search engine here
    # return documents, similarities, indices 
    query_vector = vectorizer.transform([query])
    # Transform the query into LSA space
    query_reduced = lsa.transform(query_vector)
    # Compute cosine similarities
    similarities = cosine_similarity(query_reduced, X_reduced)
    # Get the top 5 documents
    similarities = similarities.flatten()
    indices = np.argsort(similarities)[::-1][:5]
    top_similarities = similarities[indices]
    top_documents = [documents[i] for i in indices]
    return top_documents, top_similarities.tolist(), indices.tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)
    return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices}) 

if __name__ == '__main__':
    app.run(debug=True)
