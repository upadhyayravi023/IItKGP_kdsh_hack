import pathway as pw  # Retaining pathway import as requested
import os
import pandas as pd
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import pdfplumber
import numpy as np
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model and tokenizer from HuggingFace
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to generate embeddings for a given text
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings

# Placeholder: Define a mock-up of reference papers for conferences
reference_papers = {
    "CVPR": [
        "Advances in Deep Learning for Computer Vision",
        "Convolutional Neural Networks for Object Detection",
        "3D Reconstruction in Computer Vision",
        "Image Segmentation Techniques"
    ],
    "NeurIPS": [
        "Reinforcement Learning Applications in AI",
        "Generative Adversarial Networks for AI",
        "Bayesian Optimization in Machine Learning",
        "Deep Neural Networks and Their Applications"
    ],
    "EMNLP": [
        "Transformers in Natural Language Processing",
        "Text Summarization and Sentiment Analysis",
        "Multilingual NLP Techniques",
        "Advances in Natural Language Processing"
    ],
    "TMLR": [
        "Meta-Learning and Few-Shot Learning",
        "Breakthroughs in Machine Learning Research",
        "Ethics and Fairness in Machine Learning",
        "Unsupervised Learning and Clustering"
    ],
    "KDD": [
        "Innovations in Knowledge Discovery",
        "Data Mining Techniques for Large Datasets",
        "Graph Mining and Network Analysis",
        "Big Data Analytics for Business Intelligence"
    ]
}

# Function to create FAISS index for reference papers' embeddings
def create_faiss_index(reference_papers):
    embeddings = []
    labels = []
    
    for conference, papers in reference_papers.items():
        for paper in papers:
            paper_embedding = get_embedding(paper)
            embeddings.append(paper_embedding)
            labels.append(conference)
    
    embeddings = np.vstack(embeddings).astype(np.float32)
    
    # Normalize embeddings (to unit vectors) for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Build FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1])  # Use inner product (cosine similarity)
    index.add(embeddings)  # Add embeddings to the index
    
    return index, np.array(labels)

# Function to recommend conferences based on paper content using FAISS
class ConferenceSelector:
    def __init__(self, reference_data):
        self.reference_data = reference_data
        self.index, self.labels = create_faiss_index(reference_data)
    
    def recommend_conference(self, paper):
        paper_embedding = get_embedding(paper)
        paper_embedding = paper_embedding.astype(np.float32)
        
        # Normalize the query embedding (to unit vector)
        faiss.normalize_L2(paper_embedding)
        
        # Search the FAISS index (use inner product for cosine similarity)
        similarities, indices = self.index.search(paper_embedding, k=5)  # Get top 5 matches
        recommendations = []
        
        for i, idx in enumerate(indices[0]):
            conference = self.labels[idx]
            similarity = similarities[0][i]  # Cosine similarity score
            recommendations.append((conference, similarity))
        
        return sorted(recommendations, key=lambda x: x[1], reverse=True)

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# Endpoint to accept PDF and classify
@app.route('/classify', methods=['POST'])
def classify_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        paper_content = extract_text_from_pdf(file_path)
        if not paper_content:
            return jsonify({"error": "No text could be extracted from the uploaded PDF"}), 400

        selector = ConferenceSelector(reference_papers)
        recommendations = selector.recommend_conference(paper_content)

        return jsonify({
            "recommendations": [
                {"conference": rec[0], "similarity_score": rec[1]} for rec in recommendations
            ]
        })

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
