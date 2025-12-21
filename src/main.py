from summary_generator import Generator
from trl import RewardTrainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
from datasets import load_dataset
import logging
from convert_to_reward_dataset import convert_to_reward_dataset

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
    # logger.info("=== Step 2: Generating Summary pairs ===")
    # generator = Generator(
    #     model_name="mistralai/Mistral-7B-Instruct-v0.3"
    # )
    
    # Process articles and generate QA pairs
    # output_file = 'arxiv_articles.jsonl'
    # generator.generate_summary_pairs(
    #     input_file='./arxiv_articles.json',
    #     output_file=output_file
    # )
    
    # logger.info(f"QA Results saved to {output_file}")

    model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base", num_labels=1)

    dataset = load_dataset("json", data_files="arxiv_articles_labeled.jsonl")
    
    # Extract the actual dataset from DatasetDict (load_dataset returns a DatasetDict with "train" split)
    dataset = dataset["train"]
    
    dataset = convert_to_reward_dataset(dataset)

    trainer = RewardTrainer(
        model=model,
        train_dataset=dataset
    )

    trainer.train()

if __name__ == "__main__":
    main()

