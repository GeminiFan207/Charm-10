import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CharmC10NLP(nn.Module):
    def __init__(
        self,
        model_name="meta-llama/Llama-2-13b",
        freeze_encoder=False,
        use_fp16=False,
        max_new_tokens=100,
    ):
        super(CharmC10NLP, self).__init__()
        
        # Load pre-trained model and tokenizer (Causal LM for text generation)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Freeze encoder if required
        if freeze_encoder:
            self.model.eval()  # Set to eval mode
            for param in self.model.parameters():
                param.requires_grad = False
            logger.info("Model frozen for fine-tuning.")

        # Mixed precision (FP16)
        self.use_fp16 = use_fp16
        if self.use_fp16 and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed precision (FP16) enabled.")

        # Generation settings
        self.max_new_tokens = max_new_tokens
        
        # Automatically detect device (GPU/CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Model loaded on device: {self.device}")

    def generate(self, prompt, max_length=512, temperature=0.7, top_p=0.9):
        """Generates a response based on the input prompt."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=max_length,
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True  # Enable sampling for more diverse responses
            )

        # Decode the generated tokens
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response
