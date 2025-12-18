from unsloth import FastLanguageModel
from trl import SFTTrainer  # Import from trl instead
from web_scraper import ArxivScraper
from summary_generator import Generator
from transformers import AutoTokenizer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """
    Main pipeline: Scrape articles and generate QA pairs
    """
    # Step 1: Scrape articles from arXiv
    logger.info("=== Step 1: Scraping articles from arXiv ===")
    # scraper = ArxivScraper(delay=2)
    # output_file = 'arxiv_articles.json'
    
    # # Scrape articles (adjust URL as needed)
    # articles = scraper.scrape_articles_from_lists(
    #     "https://arxiv.org/list/cs/recent?skip=0&show=100",
    #     output_file=output_file
    # )
    
    # logger.info(f"Scraped {len(articles)} articles")
    
    # Step 2: Generate QA pairs
    logger.info("=== Step 2: Generating Summary pairs ===")
    generator = Generator(
        model_name="mistralai/Mistral-7B-Instruct-v0.3"
    )
    
    # Process articles and generate QA pairs
    output_file = 'arxiv_articles.jsonl'
    generator.generate_summary_pairs(
        input_file='./arxiv_articles.json',
        output_file=output_file
    )
    
    logger.info(f"QA Results saved to {output_file}")


if __name__ == "__main__":
    main()

