import json
import logging
import os
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Generator:
    """
    Question-Answer generation pipeline using Mistral-7B-Instruct model
    """
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.3", device: Optional[str] = None):
        """
        Initialize the QA Generator with Mistral model
        
        Args:
            model_name (str): Hugging Face model identifier
            device (str): Device to run the model on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        logger.info(f"Loading model: {model_name}")
        
        # Determine device
        if device is None:
            self.device = "cuda"
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def generate_summary_pairs(self, input_file: str, output_file: str) -> List[Dict[str, str]]:
        """
        Generate summary pairs from articles in a JSON file
        
        Args:
            input_file (str): Path to JSON file containing articles
            output_file (str): Path to output JSONL file
            
        Returns:
            List[Dict[str, str]]: List of dictionaries with 'response1' and 'response2' keys
        """
        try:
            # Load articles from JSON file
            with open(input_file, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            
            all_summary_pairs = []
            
            # Process each article
            for article in articles:
                article_text = article.get('article')
                if not article_text or len(article_text.strip()) == 0:
                    logger.warning("Empty article text found, skipping")
                    continue
                    
                try:
                    # Create prompt
                    prompt = self._create_prompt(article_text)
                    
                    # Tokenize
                    inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Generate
                    logger.info(f"Generating summary pairs for article...")
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=2000,
                            temperature=0.5,
                            top_p=0.9,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id
                        )
                    
                    # Decode
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Extract the generated part (after the prompt)
                    if "[/INST]" in generated_text:
                        generated_text = generated_text.split("[/INST]")[-1].strip()
                    
                    # Parse summary pairs
                    summary_pairs = self._parse_summary_from_text(generated_text)
                    if summary_pairs:
                        all_summary_pairs.append(summary_pairs)
                        
                except Exception as e:
                    logger.error(f"Error processing article: {str(e)}")
                    continue
            
            # Save all summary pairs
            self._save_summary_pairs(all_summary_pairs, output_file)
            return all_summary_pairs
            
        except Exception as e:
            logger.error(f"Error generating summary pairs: {str(e)}")
            return []
    
    def _parse_summary_from_text(self, text: str) -> Dict[str, str]:
        """
        Parse summary pairs from generated text, handling various formats

        Args:
            text (str): Generated text containing summary pairs

        Returns:
            Dict[str, str]: Parsed summary pairs as {"response1": ..., "response2": ...}
        """
        # Attempt to extract JSON using regular expressions
        json_like_pattern = r'({.*?})'
        matches = re.findall(json_like_pattern, text, re.DOTALL)
        for candidate in matches[::-1]:
            try:
                candidate = candidate.strip()
                parsed = json.loads(candidate)
                # Make sure the required keys are present
                if "response1" in parsed and "response2" in parsed:
                    return parsed
            except Exception:
                continue

        # If direct JSON doesn't work, try to construct manually
        # Find lines containing 'response1' and 'response2'
        responses = {}
        pattern = r'"?(response[12])"?\s*:\s*"?(.+?)"?[\n,}]'
        matches = re.findall(pattern, text)
        for key, value in matches:
            responses[key] = value.strip().strip('"')
        if "response1" in responses and "response2" in responses:
            return responses

        # Fallback: try extracting two non-empty lines (less robust, but last resort)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        found = []
        for line in lines:
            # Accept up to two significant (longer) lines
            if len(line) > 10 and len(found) < 2:
                found.append(line)
        if len(found) == 2:
            return {"response1": found[0], "response2": found[1]}

        # If all parsing fails
        logger.warning("Could not parse summary pairs from text: %s", text)
        return {}
        
    def _create_prompt(self, article_text: str) -> str:
        """
        Create a prompt for summary generation
        
        Args:
            article_text (str): The article content
            
        Returns:
            str: Formatted prompt
        """
        # Truncate article if too long (to fit in context window)
        max_length = 2000  # Adjust based on model context window
        if len(article_text) > max_length:
            article_text = article_text[:max_length] + "..."
        
        prompt = f"""<s>[INST] You are an expert at summarizing academic articles. 
                    Generate two precise and meaningful summaries around 150 words on the following article content.
                    Format your response as a JSON object where the object has "response1" and "response2" fields.

                    Article content:
                    {article_text}

                    Generate the pairs in JSON format: [/INST]"""
        
        return prompt

    def _save_summary_pairs(self, summary: List[Dict[str, str]], filename: str):
        """
        Save summary pairs to a JSONL file.

        Args:
            summary (List[Dict[str, str]]): List of summary pair dicts, each with "response1" and "response2".
            filename (str): Output filename
        """
        try:
            with open(filename, "w", encoding='utf-8') as outfile:
                for item in summary:
                    # Save the summary pairs directly as JSON
                    json.dump(item, outfile, ensure_ascii=False)
                    outfile.write('\n')
        except Exception as e:
            logger.error(f"Error saving summary pairs to {filename}: {str(e)}")
