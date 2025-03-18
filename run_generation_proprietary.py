import os
import json
import re
import numpy as np
import logging
import asyncio
import time
from tqdm.asyncio import tqdm_asyncio
from datasets import load_dataset
from openai import AsyncOpenAI


# Model name
# MODEL_NAME = "gpt-4o"
MODEL_NAME="gemini-1.5-pro"
# MODEL_NAME="deepseek-chat"

# Dataset details
DATASET_PATH = "ikim-uk-essen/Note2FHIR"
TEST_SPLIT = "test"

# Maximum concurrent requests
MAX_CONCURRENT = 20

# Prompt template
PROMPT_TEMPLATE = (
    "Given the following patient note, generate the corresponding FHIR JSON representation. "
    "Please ensure that:\\n "
    "- All keys and string values are enclosed in double quotes.\\n "
    "- The JSON is valid and follows proper syntax.\\n "
    "- There are no comments or code expressions.\\n "
    "- Enclose the JSON output within triple backticks with 'json' specified, like this:\\n\\n "
    "json\\n "
    "<Your JSON here>\\n "
    "\\n\\n "
    "Make sure the first resourceType is Bundle.\\n\\n "
    "Patient Note:\\n "
    "{{note}}\\n\\n "
    "Generate FHIR JSON:"
)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_json(text):
    """
    Extracts JSON from text, including from code blocks.
    Handles both ```json and regular JSON objects.
    """
    # First try to extract from code blocks
    code_block_pattern = re.compile(r'```(?:json)?\n(.*?)\n```', re.DOTALL)
    match = code_block_pattern.search(text)
    
    if match:
        json_str = match.group(1).strip()
        return json.loads(json_str)
    else:
        raise ValueError("No JSON block found in the text.")


def flatten_json(y, prefix=''):
    """Flattens a nested JSON into a dictionary of dot-delimited paths -> values."""
    items = {}
    if isinstance(y, dict):
        for k, v in y.items():
            items.update(flatten_json(v, f"{prefix}{k}."))   
    elif isinstance(y, list):
        for i, v in enumerate(y):
            items.update(flatten_json(v, f"{prefix}{i}."))   
    else:
        items[prefix.rstrip('.')] = y
    return items

def compute_similarity(ground_truth, prediction):
    """Computes the F1 score between the flattened ground_truth and prediction JSONs."""
    gt_flat = flatten_json(ground_truth)
    pred_flat = flatten_json(prediction)
    gt_items = set(gt_flat.items())
    pred_items = set(pred_flat.items())

    true_positives = gt_items & pred_items
    false_positives = pred_items - gt_items
    false_negatives = gt_items - pred_items

    precision = (
        len(true_positives) / (len(true_positives) + len(false_positives))
        if pred_items else 0
    )
    
    recall = (
        len(true_positives) / (len(true_positives) + len(false_negatives))
        if gt_items else 0
    )
    
    if (precision + recall) == 0:
        f1_score = 0
    else:
        f1_score = 2 * precision * recall / (precision + recall)

    return f1_score

def calculate_similarity_scores(references, predictions):
    """Compute similarity scores between reference and predicted FHIR JSONs."""
    scores = []
    for i, (ref_str, pred_str) in enumerate(zip(references, predictions)):
        try:
            fhir_true = json.loads(ref_str)
            fhir_pred = extract_json(pred_str)
            similarity_score = compute_similarity(fhir_true, fhir_pred)
            scores.append(similarity_score)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Error processing example {i}: {e}")
            scores.append(0.0)

    return scores


def doc_to_text(example):
    """Build the prompt from the example's 'note' field."""
    return PROMPT_TEMPLATE.replace("{{note}}", example["note"])

def doc_to_target(example):
    """Get reference FHIR JSON string."""
    return example["fhir"]

async def get_completion(client, prompt, semaphore):
    """Get GPT completion with error handling and retries."""
    async with semaphore:
        system_prompt = "You are a FHIR expert that can check if the generated fhir json are correct."
        
        for attempt in range(3):  # Simple retry mechanism
            try:
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt < 2:  # Don't sleep after the last attempt
                    delay = 1 * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"API error (attempt {attempt+1}/3): {str(e)}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Failed after 3 attempts: {str(e)}")
                    return None

async def process_example(client, example, semaphore):
    """Process a single example asynchronously."""
    prompt = doc_to_text(example)
    prediction = await get_completion(client, prompt, semaphore)
    
    if prediction is None:
        return {
            "reference": doc_to_target(example),
            "prediction": "",
            "error": True
        }
    
    return {
        "reference": doc_to_target(example),
        "prediction": prediction,
        "error": False
    }


async def evaluate_fhir_generation_async(dataset_path=DATASET_PATH, test_split=TEST_SPLIT, batch_size=None):
    """Evaluate FHIR generation using async processing."""
    # Load dataset
    logger.info(f"Loading dataset '{dataset_path}' split='{test_split}'")
    dataset = load_dataset(dataset_path, split=test_split)
    
    # Limit batch size if specified
    if batch_size:
        dataset = dataset.select(range(min(batch_size, len(dataset))))
    
    # Initialize API client
    # client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    client = AsyncOpenAI(
    api_key="KEY",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
    # client = AsyncOpenAI(api_key="KEY", base_url="https://api.deepseek.com")
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    # Start timing
    start_time = time.time()
    logger.info(f"Starting evaluation on {len(dataset)} examples with {MAX_CONCURRENT} concurrent requests")
    
    # Create tasks for all examples
    tasks = [process_example(client, example, semaphore) for example in dataset]
    
    # Use tqdm_asyncio.gather instead of wrapping asyncio.gather with tqdm
    results = await tqdm_asyncio.gather(*tasks, total=len(tasks), desc="Evaluating")
    
    # Extract references and predictions
    references = [r["reference"] for r in results]
    predictions = [r["prediction"] for r in results]
    failed_count = sum(1 for r in results if r["error"])
    
    # Calculate similarity scores
    logger.info("Computing similarity scores...")
    scores = calculate_similarity_scores(references, predictions)
    mean_score = np.mean(scores) if scores else 0.0
    
    # Log results
    elapsed = time.time() - start_time
    logger.info(f"Evaluation completed in {elapsed:.2f} seconds")
    logger.info(f"Successful evaluations: {len(dataset) - failed_count}/{len(dataset)}")
    logger.info(f"FHIR Similarity Score (F1): {mean_score:.4f}")
    
    return {
        "similarity_score": mean_score,
        "individual_scores": scores,
        "elapsed_time": elapsed,
        "total_examples": len(dataset),
        "failed_examples": failed_count
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate FHIR generation with GPT")
    parser.add_argument("--dataset", type=str, default=DATASET_PATH, help="Dataset path")
    parser.add_argument("--split", type=str, default=TEST_SPLIT, help="Dataset split")
    parser.add_argument("--batch-size", type=int, default=None, help="Limit evaluation to N examples")
    parser.add_argument("--concurrent", type=int, default=MAX_CONCURRENT, help="Max concurrent requests")
    
    args = parser.parse_args()
    MAX_CONCURRENT = args.concurrent
    
    # Run the async evaluation
    asyncio.run(evaluate_fhir_generation_async(
        dataset_path=args.dataset,
        test_split=args.split,
        batch_size=args.batch_size
    ))
