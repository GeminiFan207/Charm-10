import onnxruntime as ort
import numpy as np
import torch
import torchaudio
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM
from typing import Union, List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InferenceEngine:
    def __init__(self, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the InferenceEngine.

        Args:
            model_path (str): Path to the ONNX model.
            device (str): Device to run the model on ("cuda" or "cpu").
        """
        self.device = device
        try:
            # Initialize ONNX runtime session
            self.session = ort.InferenceSession(
                model_path,
                providers=[
                    "TensorrtExecutionProvider",
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider"
                ]
            )
            logger.info(f"ONNX model loaded successfully on device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise

    def run_text_inference(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str, max_length: int = 200) -> str:
        """
        Run text inference using a causal language model.

        Args:
            model (AutoModelForCausalLM): Pre-trained causal language model.
            tokenizer (AutoTokenizer): Tokenizer for the model.
            prompt (str): Input text prompt.
            max_length (int): Maximum length of the generated text.

        Returns:
            str: Generated text.
        """
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = model.generate(**inputs, max_length=max_length)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Text inference failed: {e}")
            raise

    def run_image_inference(self, clip_model: CLIPModel, processor: CLIPProcessor, image_path: str) -> np.ndarray:
        """
        Run image inference using a CLIP model.

        Args:
            clip_model (CLIPModel): Pre-trained CLIP model.
            processor (CLIPProcessor): Processor for the CLIP model.
            image_path (str): Path to the input image.

        Returns:
            np.ndarray: Image features as a numpy array.
        """
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(self.device)
            outputs = clip_model.get_image_features(**inputs)
            return outputs.cpu().detach().numpy()
        except Exception as e:
            logger.error(f"Image inference failed: {e}")
            raise

    def run_audio_inference(self, whisper_model: Any, audio_file: str) -> str:
        """
        Run audio inference using a Whisper model.

        Args:
            whisper_model (Any): Pre-trained Whisper model.
            audio_file (str): Path to the input audio file.

        Returns:
            str: Transcribed text.
        """
        try:
            waveform, sample_rate = torchaudio.load(audio_file)
            waveform = waveform.to(self.device)
            return whisper_model.transcribe(waveform)["text"]
        except Exception as e:
            logger.error(f"Audio inference failed: {e}")
            raise

    def run_general_inference(self, input_data: Union[np.ndarray, List, Dict]) -> np.ndarray:
        """
        Run general inference using the ONNX model.

        Args:
            input_data (Union[np.ndarray, List, Dict]): Input data for the model.

        Returns:
            np.ndarray: Model output.
        """
        try:
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name

            # Ensure input_data is a numpy array
            if not isinstance(input_data, np.ndarray):
                input_data = np.array(input_data, dtype=np.float32)

            return self.session.run([output_name], {input_name: input_data})[0]
        except Exception as e:
            logger.error(f"General inference failed: {e}")
            raise