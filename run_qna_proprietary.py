import os
import logging
import asyncio
import time
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI
from datasets import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MODEL = "gpt-4.5-preview"
# MODEL= "deepseek-chat"
# MODEL= "gemini-2.0-flash"
MAX_CONCURRENT = 10

def doc_to_text(example):
    """Format example as a prompt."""
    return (
        f"Question: {example['question']}\n"
        f"Options:\n"
        f"A: {example['mc_answer1']}\n"
        f"B: {example['mc_answer2']}\n"
        f"C: {example['mc_answer3']}\n"
        f"D: {example['mc_answer4']}\n"
        "Answer:"
    )

def doc_to_target(example):
    """Get the correct answer index."""
    return ["1", "2", "3", "4"].index(example["correct_answer_num"])

async def get_response(client, prompt, semaphore):
    """Get GPT-4 response with error handling and retry logic."""
    async with semaphore:
        for attempt in range(3):  # Simple retry mechanism
            try:
                completion = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful model that only responds with one of [A, B, C, D]."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=1
                )
                return completion.choices[0].message.content.strip()
            except Exception as e:
                if attempt < 2:  # Don't sleep after the last attempt
                    await asyncio.sleep(1 * (2 ** attempt))  # Simple exponential backoff
                else:
                    logger.warning(f"Failed after 3 attempts: {str(e)}")
                    return None

async def process_example(client, example, semaphore):
    """Process a single example."""
    prompt = doc_to_text(example)
    predicted = await get_response(client, prompt, semaphore)
    
    # Handle failed API calls
    if predicted is None:
        return {"error": True}
        
    choices = ["A", "B", "C", "D"]
    correct_index = doc_to_target(example)
    correct_choice = choices[correct_index]
    is_correct = predicted == correct_choice
    
    return {
        "is_correct": is_correct,
        "predicted": predicted,
        "correct": correct_choice
    }

async def evaluate_dataset(dataset_path="ikim-uk-essen/FHIR-QA", split="test", batch_size=None):
    """Evaluate GPT-4 on dataset using async processing."""
    # Load dataset
    logger.info(f"Loading dataset: {dataset_path}")
    dataset = load_dataset(dataset_path, split=split)
    
    # Limit batch size if specified
    if batch_size:
        dataset = dataset.select(range(min(batch_size, len(dataset))))
        
    # Initialize API client
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    # client = AsyncOpenAI(
    # api_key="KEY",
    # base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    # )
    # client = AsyncOpenAI(api_key="KEY", base_url="https://api.deepseek.com")
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    # Start timer
    start_time = time.time()
    logger.info(f"Starting evaluation on {len(dataset)} examples with {MAX_CONCURRENT} concurrent requests")
    
    # Create tasks for all examples
    tasks = [process_example(client, example, semaphore) for example in dataset]
    
    # Execute tasks with progress bar
    results = await tqdm.gather(*tasks, desc="Evaluating")
    
    # Calculate metrics
    valid_results = [r for r in results if "error" not in r]
    correct_count = sum(1 for r in valid_results if r["is_correct"])
    
    # Log results
    elapsed = time.time() - start_time
    logger.info(f"Evaluation completed in {elapsed:.2f} seconds")
    logger.info(f"Successfully evaluated: {len(valid_results)}/{len(dataset)} examples")
    logger.info(f"Accuracy: {correct_count/len(valid_results):.4f}")
    
    return {
        "accuracy": correct_count/len(valid_results) if valid_results else 0,
        "total": len(dataset),
        "successful": len(valid_results),
        "time": elapsed
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate GPT-4 on multiple-choice questions")
    parser.add_argument("--dataset", type=str, default="ikim-uk-essen/FHIR-QA", help="Dataset path") # Change to ikim-uk-essen/FHIR-RESTQA for REST QA
    parser.add_argument("--split", type=str, default="test", help="Dataset split") 
    parser.add_argument("--batch-size", type=int, default=None, help="Limit evaluation to N examples")
    parser.add_argument("--concurrent", type=int, default=MAX_CONCURRENT, help="Max concurrent requests")
    
    args = parser.parse_args()
    MAX_CONCURRENT = args.concurrent
    
    asyncio.run(evaluate_dataset(
        dataset_path=args.dataset,
        split=args.split,
        batch_size=args.batch_size
    ))