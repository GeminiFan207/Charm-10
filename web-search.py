import os
import logging
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import faiss  # For efficient similarity search

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Firebase Admin SDK
firebase_cred_path = os.getenv("FIREBASE_CRED_PATH", "path/to/your/firebase/credentials.json")
cred = credentials.Certificate(firebase_cred_path)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Initialize the model for embeddings
model = SentenceTransformer('all-mpnet-base-v2')  # Better model for semantic search

# Initialize FAISS index for efficient similarity search
embedding_size = 768  # Size of embeddings from 'all-mpnet-base-v2'
faiss_index = faiss.IndexFlatIP(embedding_size)  # Inner product for cosine similarity
document_ids = []  # To map FAISS index to document IDs

# Initialize Flask app
app = Flask(__name__)

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Function to store documents in Firebase and update FAISS index
def store_documents(documents):
    try:
        batch = db.batch()
        embeddings = model.encode(documents)  # Encode all documents at once

        for idx, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_ref = db.collection('documents').document(f'doc_{idx}')
            batch.set(doc_ref, {
                'text': doc,
                'embedding': embedding.tolist()
            })
            document_ids.append(doc_ref.id)  # Store document ID
            faiss_index.add(np.array([embedding]))  # Add embedding to FAISS index

        batch.commit()
        logger.info(f"Stored {len(documents)} documents in Firebase and updated FAISS index.")
    except Exception as e:
        logger.error(f"Error storing documents: {e}")

# Function to search documents using FAISS
def search_documents(query):
    try:
        query_embedding = model.encode(query).reshape(1, -1)
        distances, indices = faiss_index.search(query_embedding, k=1)  # Find top-1 match

        if indices[0][0] == -1:
            return None  # No match found

        # Retrieve the best matching document from Firebase
        best_doc_id = document_ids[indices[0][0]]
        best_doc = db.collection('documents').document(best_doc_id).get()
        return best_doc.to_dict()
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        return None

@app.route('/search', methods=['GET'])
@limiter.limit("10 per minute")  # Rate limit for search endpoint
def search():
    query = request.args.get('query')
    if not query:
        return jsonify({"error": "No query provided."}), 400

    result = search_documents(query)
    if result:
        return jsonify(result)
    else:
        return jsonify({"error": "No relevant documents found."}), 404

if __name__ == '__main__':
    # Example: Insert your documents to Firebase (use your own content)
    documents = [
        "Your custom document content goes here.",
        "Another example document for semantic search.",
        "This is a test document for the AI search system."
    ]
    store_documents(documents)

    # Run Flask app for the AI search
    app.run(debug=True)
