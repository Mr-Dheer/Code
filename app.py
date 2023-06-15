# from flask import Flask, render_template, request
# from nltk.tokenize import sent_tokenize
# from sklearn.feature_extraction.text import TfidfVectorizer

# app = Flask(__name__)


# def extractive_summarization(text, num_sentences):
#     # Tokenize the text into sentences
#     sentences = sent_tokenize(text)

#     # Calculate TF-IDF scores for the sentences
#     tfidf_vectorizer = TfidfVectorizer()
#     sentence_scores = tfidf_vectorizer.fit_transform(sentences)

#     # Calculate sentence importance scores based on TF-IDF scores
#     sentence_importance = sentence_scores.sum(axis=1)

#     # Sort sentences by their importance scores in descending order
#     ranked_sentences = [sentence for _, sentence in sorted(
#         zip(sentence_importance, sentences), reverse=True)]

#     # Select the top N important sentences
#     selected_sentences = ranked_sentences[:num_sentences]

#     # Combine the selected sentences to form the extractive summary
#     extractive_summary = " ".join(selected_sentences)

#     return extractive_summary


# @app.route('/')
# def index():
#     return render_template('index.html')


# @app.route('/', methods=['POST'])
# def summarize():
#     text = request.form['text']
#     sentences_count = int(request.form['sentences_count'])

#     summary = extractive_summarization(text, sentences_count)

#     return render_template('index.html', summary=summary)


# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request
from nltk.tokenize import sent_tokenize
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer


app = Flask(__name__)


def generate_summary(text, num_sentences=3):
    sentences = sent_tokenize(text)
    tfidf_vectorizer = TfidfVectorizer()
    sentence_scores = tfidf_vectorizer.fit_transform(sentences)
    similarity_matrix = (sentence_scores * sentence_scores.T).toarray()

    # Build the graph from the similarity matrix
    graph = nx.from_numpy_array(similarity_matrix)

    # Apply the PageRank algorithm
    scores = nx.pagerank(graph)

    # Sort the sentences by their scores
    ranked_sentences = sorted(
        ((scores[i], sentence) for i, sentence in enumerate(sentences)), reverse=True)

    # Select the top sentences as the summary
    summary = ' '.join(
        [sentence for _, sentence in ranked_sentences[:num_sentences]])

    return summary


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def summarize():
    text = request.form['text']
    sentences_count = int(request.form['sentences_count'])

    summary = generate_summary(text, sentences_count)

    return render_template('index.html', summary=summary)


if __name__ == '__main__':
    app.run(debug=True)
