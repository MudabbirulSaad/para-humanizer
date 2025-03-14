"""
Core paraphraser module for UltimateParaphraser.
Provides the main UltimateParaphraser class for orchestrating text paraphrasing.
"""
import torch
import concurrent.futures
import time
import logging
from typing import List, Dict, Tuple, Any, Optional, Union
import random
from para_humanizer.utils.config_manager import get_config_manager
from para_humanizer.utils.hardware import detect_gpu, optimize_torch_settings
from para_humanizer.utils.text_utils import chunk_text, fix_formatting, preserve_structure_paraphrase
from para_humanizer.processors.rule_based import RuleBasedProcessor
from para_humanizer.processors.humanizer import Humanizer
from para_humanizer.processors.transformer import TransformerProcessor
from para_humanizer.utils.synonym_loader import get_synonym_library
from para_humanizer.utils.synonym_learner import get_synonym_learner

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UltimateParaphraser:
    """
    Main paraphraser class that coordinates between different processors.
    Provides a unified interface for various paraphrasing approaches.
    """
    
    def __init__(self, use_gpu: bool = True, batch_size: int = 8, 
                 hybrid_mode: bool = True, transformer_disable: bool = False,
                 enable_learning: bool = True):
        """
        Initialize the paraphraser with the specified configuration.
        
        Args:
            use_gpu: Whether to use GPU acceleration if available
            batch_size: Batch size for processing
            hybrid_mode: Whether to use both rule-based and transformer approaches
            transformer_disable: Force disabling transformer models
            enable_learning: Whether to enable automatic synonym learning
        """
        self.use_gpu = use_gpu and detect_gpu()
        self.batch_size = batch_size
        self.hybrid_mode = hybrid_mode
        
        # If Windows or macOS and transformer_disable not explicit, default to disabling transformers
        # This is due to common compatibility issues on these platforms
        if transformer_disable is None:
            import platform
            transformer_disable = platform.system() in ['Windows', 'Darwin']
            
        self.transformer_disable = transformer_disable
        self.enable_learning = enable_learning
        
        # Initialize configuration manager
        self.config_manager = get_config_manager()
        
        # Load configuration
        blacklist_words = set(self.config_manager.get_blacklist_words())
        common_words = set(self.config_manager.get_set('common_words'))
        
        # Initialize synonym library using existing functions
        self.synonym_library = get_synonym_library(blacklist_words, common_words)
        
        # Initialize processors with configuration
        self.rule_processor = RuleBasedProcessor(
            config_manager=self.config_manager,
            synonym_library=self.synonym_library,
            enable_learning=enable_learning
        )
        self.humanizer = Humanizer()
        
        # Initialize transformer processor if not disabled
        self.transformer_processor = None
        if not self.transformer_disable:
            try:
                # Convert use_gpu to the appropriate device string
                device = "cuda" if self.use_gpu else "cpu"
                self.transformer_processor = TransformerProcessor(
                    device=device
                )
            except Exception as e:
                print(f"Warning: Failed to initialize transformer processor: {e}")
                print("Falling back to rule-based paraphrasing only.")
                
        # Log configuration
        print(f"UltimateParaphraser initialized:")
        print(f"  - GPU: {'Enabled' if self.use_gpu else 'Disabled'}")
        print(f"  - Batch size: {self.batch_size}")
        print(f"  - Hybrid mode: {'Enabled' if self.hybrid_mode else 'Disabled'}")
        print(f"  - Transformer models: {'Disabled' if self.transformer_disable else 'Enabled'}")
        print(f"  - Automatic synonym learning: {'Enabled' if self.enable_learning else 'Disabled'}")
        
    def paraphrase(self, text: str, rule_based_rate: float = 0.4, transformer_rate: float = 0.0,
                   humanize: bool = True, humanize_intensity: float = 0.5, 
                   typo_rate: float = 0.0, no_parallel: bool = False,
                   preserve_structure: bool = False, tone: str = "casual") -> str:
        """
        Paraphrase the input text using the configured processors.
        
        Args:
            text: Input text to paraphrase
            rule_based_rate: Rate of word replacement for rule-based (0.0 to 1.0)
            transformer_rate: Rate of transformer usage (0.0 to 1.0)
            humanize: Whether to apply humanization
            humanize_intensity: Intensity of humanization (0.0 to 1.0)
            typo_rate: Rate of typo introduction (0.0 to 1.0)
            no_parallel: Whether to disable parallel processing
            preserve_structure: Whether to preserve the original document structure
            tone: Text tone - "casual", "formal", or "academic"
            
        Returns:
            Paraphrased text
        """
        # Validation
        rule_based_rate = max(0.0, min(1.0, rule_based_rate))
        transformer_rate = max(0.0, min(1.0, transformer_rate))
        humanize_intensity = max(0.0, min(1.0, humanize_intensity))
        typo_rate = max(0.0, min(1.0, typo_rate))
        
        # Early exit for empty input
        if not text or not text.strip():
            return text
            
        # Core processing variables
        result_text = text
        start_time = time.time()
        
        # For academic tone, automatically apply preserve_structure
        if tone == "academic" and not preserve_structure:
            preserve_structure = True
            print("Setting preserve_structure=True for academic tone")
        
        # Handle structure preservation
        if preserve_structure:
            result_text = preserve_structure_paraphrase(
                text, 
                lambda chunk: self._process_text(
                    chunk, rule_based_rate, transformer_rate, 
                    humanize, humanize_intensity, typo_rate, no_parallel, tone
                )
            )
        else:
            # Normal processing without structure preservation
            result_text = self._process_text(
                text, rule_based_rate, transformer_rate, 
                humanize, humanize_intensity, typo_rate, no_parallel, tone
            )
        
        # Log time taken
        time_taken = time.time() - start_time
        logger.info(f"Paraphrasing completed in {time_taken:.2f} seconds")
            
        return result_text
    
    def _process_text(self, text: str, rule_based_rate: float, transformer_rate: float,
                     humanize: bool, humanize_intensity: float, typo_rate: float, 
                     no_parallel: bool, tone: str = "casual") -> str:
        """
        Internal method to process text with all enabled processors.
        
        Args:
            text: Input text to process
            rule_based_rate: Rate of word replacement (0.0 to 1.0)
            transformer_rate: Rate of transformer usage (0.0 to 1.0)
            humanize: Whether to apply humanization
            humanize_intensity: Intensity of humanization (0.0 to 1.0)
            typo_rate: Rate of typo introduction (0.0 to 1.0)
            no_parallel: Whether to disable parallel processing
            tone: Text tone - "casual", "formal", or "academic"
            
        Returns:
            Processed text
        """
        # Early return for empty text
        if not text or not text.strip():
            return text
        
        # Apply rule-based processing
        if rule_based_rate > 0:
            text = self.rule_processor.process_text(text, rule_based_rate, tone=tone)
        
        # Apply transformer processing if available and requested
        if transformer_rate > 0 and not self.transformer_disable and self.transformer_processor:
            text = self.transformer_processor.process_text(text, transformer_rate)
            
        # Apply humanization if requested
        if humanize and humanize_intensity > 0:
            text = self.humanizer.humanize(text, humanize_intensity, typo_rate, tone)
            
        return text
        
    def _paraphrase_preserving_structure(self, text: str, rule_based_rate: float = 0.4,
                                        transformer_rate: float = 0.0, humanize: bool = True,
                                        humanize_intensity: float = 0.5, typo_rate: float = 0.0) -> str:
        """
        Paraphrase text while preserving its original structure.
        
        Args:
            text: Input text to paraphrase
            rule_based_rate: Rate of word replacement for rule-based (0.0 to 1.0)
            transformer_rate: Rate of transformer usage (0.0 to 1.0)
            humanize: Whether to apply humanization
            humanize_intensity: Intensity of humanization (0.0 to 1.0)
            typo_rate: Rate of typo introduction (0.0 to 1.0)
            
        Returns:
            Paraphrased text with preserved structure
        """
        # Define a function to process individual text chunks
        def process_chunk(chunk_text: str) -> str:
            result = chunk_text
            
            # Apply rule-based paraphrasing if enabled
            if rule_based_rate > 0:
                result = self.rule_processor.paraphrase_text(result, rate=rule_based_rate)
                
            # Apply transformer if enabled and available
            if self.transformer_processor and transformer_rate > 0:
                if random.random() < transformer_rate:
                    try:
                        result = self.transformer_processor.paraphrase(result)
                    except Exception as e:
                        print(f"Transformer error on chunk: {e}")
                        
            # Apply humanization if enabled
            if humanize and humanize_intensity > 0:
                result = self.humanize(result, intensity=humanize_intensity, typo_rate=typo_rate)
                
            return result
            
        # Use the preserve_structure_paraphrase utility
        return preserve_structure_paraphrase(text, process_chunk)
        
    def humanize(self, text: str, intensity: float = 0.5, typo_rate: float = 0.005) -> str:
        """
        Apply humanization techniques to the text without paraphrasing.
        
        Args:
            text: Input text to humanize
            intensity: Intensity of humanization (0.0 to 1.0)
            typo_rate: Rate of typo introduction (0.0 to 1.0)
            
        Returns:
            Humanized text
        """
        if not text or not text.strip():
            return text
            
        return self.humanizer.humanize(text, intensity=intensity, typo_rate=typo_rate)
        
    def get_learning_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the automatic synonym learning.
        
        Returns:
            Dictionary with learning statistics
        """
        if self.enable_learning:
            return self.rule_processor.get_learning_stats()
        return {"enabled": False, "message": "Automatic synonym learning is disabled"}
        
    def reset_learning_stats(self) -> None:
        """
        Reset the synonym learning statistics.
        """
        if self.enable_learning:
            self.rule_processor.reset_learning()
            print("Synonym learning statistics have been reset.")
        else:
            print("Automatic synonym learning is disabled.")
            
    def batch_paraphrase(self, texts: List[str], **kwargs) -> List[str]:
        """
        Paraphrase a batch of texts with the same settings.
        
        Args:
            texts: List of texts to paraphrase
            **kwargs: Arguments to pass to paraphrase method
            
        Returns:
            List of paraphrased texts
        """
        return [self.paraphrase(text, **kwargs) for text in texts]
