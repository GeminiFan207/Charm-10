import torch
import torch.nn as nn
from transformers import AutoTokenizer
import firebase_admin
from firebase_admin import credentials, firestore
import logging

# Import classes and functions from model.py
from inference.model import ModelArgs, Transformer

# Setup Firebase
cred = credentials.Certificate("firebase-config.json")  # Replace with your Firebase Admin SDK JSON
firebase_admin.initialize_app(cred)
db = firestore.client()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CharmC10NLP(nn.Module):
    def __init__(
        self,
        model_name="meta-llama/Llama-2-13b",
        freeze_encoder=False,
        use_fp16=False,
        max_new_tokens=200,
    ):
        super(CharmC10NLP, self).__init__()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize ModelArgs
        self.args = ModelArgs()
        
        # Initialize Transformer model
        self.model = Transformer(self.args)
        
        # Freeze the model if required
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
            logger.info("Model frozen for fine-tuning.")

        # Mixed precision (FP16)
        self.use_fp16 = use_fp16
        if self.use_fp16 and torch.cuda.is_available():
            logger.info("FP16 enabled for mixed precision training.")

        # Generation settings
        self.max_new_tokens = max_new_tokens
        
        # Auto-detect device (GPU/CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Model running on {self.device}")

    def generate_response(self, prompt, temperature=0.7, top_p=0.9):
        """Generates an AI response based on input prompt."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )

        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Store AI response in Firestore
        self.store_response(prompt, response)
        
        return response

    def store_response(self, prompt, response):
        """Saves the AI-generated response to Firestore."""
        doc_ref = db.collection("chatflare_responses").document()
        doc_ref.set({
            "prompt": prompt,
            "response": response,
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        logger.info("Response stored in Firestore.")