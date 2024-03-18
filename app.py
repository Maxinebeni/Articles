from flask import Flask, render_template, request
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import nltk
nltk.download('punkt')


nltk.download('stopwords')


app = Flask(__name__)

# Load TF-IDF vectorizer and cosine similarity matrix
with open('cosine_model.pkl', 'rb') as file:
    tfidf_vectorizer_loaded, cosine_similarity_matrix_loaded = pickle.load(file)

# Load DataFrame containing health articles only
df_health = pd.read_csv("news_articles.csv")
df_health = df_health[df_health['category'] == 'health']

# Preprocessing setup
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    # Tokenize text
    tokens = word_tokenize(text.lower())
    # Remove stopwords and non-alphabetic tokens, and stem words
    tokens = [stemmer.stem(word) for word in tokens if word.isalpha() and word not in stop_words]
    # Join tokens back into a single string
    return ' '.join(tokens)

# Apply preprocessing to 'body' column
df_health['processed_body'] = df_health['body'].apply(preprocess)

# Vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df_health['processed_body'])

# Placeholder article for empty recommendations
placeholder_article = {
    'title': 'Contributing to collaborative health governance in Africa: a realist evaluation of the Universal Health Coverage Partnership',
    'url': 'https://link.springer.com/article/10.1186/s12913-022-08120-0'
}

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    if request.method == 'POST':
        summary = request.form['summary']
        processed_summary = preprocess(summary)
        summary_vectorized = tfidf_vectorizer.transform([processed_summary])
        similarity_scores = cosine_similarity(summary_vectorized, tfidf_matrix)
        similar_indices = similarity_scores.argsort()[0][::-1]
        similar_articles = df_health.iloc[similar_indices]
        similar_articles = similar_articles[similarity_scores[0][similar_indices] > 0.10][:2]
        if len(similar_articles) == 0:
            recommendations.append(placeholder_article)
        else:
            recommendations.extend(similar_articles['url'].tolist())
        return render_template('index.html', recommendations=recommendations)
    else:
        return render_template('index.html', recommendations=None)

if __name__ == '__main__':
    app.run(debug=True)
