import os
import time
import torch
import logging
import soundfile as sf
from kokoro import KPipeline
from config import settings

logger = logging.getLogger(__name__)

class TTSService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() and settings.USE_GPU else "cpu"
        self.pipeline = None
        self.audio_cache_dir = os.path.join(settings.MONITORED_ROOT, ".sefs_cache", "audio")
        
        if not os.path.exists(self.audio_cache_dir):
            os.makedirs(self.audio_cache_dir, exist_ok=True)
            
        logger.info(f"TTS Service initialized on {self.device}. Cache dir: {self.audio_cache_dir}")

    def _load_model(self):
        if self.pipeline is None:
            logger.info("Loading Kokoro TTS model...")
            try:
                # lang_code='a' is for American English
                self.pipeline = KPipeline(lang_code='a', device=self.device)
                logger.info("Kokoro TTS model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load Kokoro TTS model: {e}")
                raise

    def generate_audio(self, text: str, file_hash: str) -> str:
        """
        Generates audio for the given text.
        Returns the path to the generated audio file.
        """
        if not text:
            raise ValueError("No text provided for TTS")

        # Create a deterministic filename based on file hash and text length (simple versioning)
        # We use text length to invalidate cache if summary changes but file hash is same (edge case)
        # Better: hash the text itself for the audio filename
        text_hash = str(hash(text) % 1000000)
        filename = f"{file_hash}_{text_hash}.wav"
        output_path = os.path.join(self.audio_cache_dir, filename)

        # Check cache
        if os.path.exists(output_path):
            logger.info(f"TTS Cache hit: {filename}")
            return output_path

        self._load_model()

        logger.info(f"Generating audio for hash {file_hash}...")
        try:
            generator = self.pipeline(text, voice='af_heart', speed=1, split_pattern=r'\n+')
            
            # Combine all audio segments
            all_audio = []
            for i, (gs, ps, audio) in enumerate(generator):
                all_audio.append(audio)
            
            if not all_audio:
                raise ValueError("No audio generated")
                
            import numpy as np
            final_audio = np.concatenate(all_audio)
            
            sf.write(output_path, final_audio, 24000)
            logger.info(f"Audio saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"TTS Generation failed: {e}")
            raise
