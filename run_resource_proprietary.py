import os
import logging
import asyncio
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI
from datasets import load_dataset
import huggingface_hub
from sklearn.metrics import f1_score
import argparse
import numpy as np

# Default configuration
DATASET_PATH = "ikim-uk-essen/FHIR-ResourceID"
TEST_SPLIT = "test"
MAX_CONCURRENT = 10
MODEL_NAME="deepseek-chat"
# MODEL_NAME="gpt-4o"
# MODEL_NAME="gemini-2.0-flash"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluate_async")

# Define the list of valid resource choices
DOC_TO_CHOICE = [
    "Immunization", "ImagingStudy", "Patient", "AllergyIntolerance", 
    "SupplyDelivery", "CareTeam", "Procedure", "DiagnosticReport", 
    "ExplanationOfBenefit", "Condition", "Practitioner", "Organization", 
    "Encounter", "MedicationAdministration", "Observation", 
    "MedicationRequest", "CarePlan"
]

# Build prompt with the required format
def doc_to_text(example):
    return (
        f"{example['resource']}\n"
        "Question: What type of FHIR resource is this based on its structure?\n"
        "Possible resource types (please choose exactly one):\n"
        "Immunization, ImagingStudy, Patient, AllergyIntolerance, SupplyDelivery, CareTeam, Procedure,\n"
        "DiagnosticReport, ExplanationOfBenefit, Condition, Practitioner, Organization, Encounter,\n"
        "MedicationAdministration, Observation, MedicationRequest, CarePlan\n"
        "Answer:"
    )

# Convert gold label to index
def doc_to_target(example):
    return DOC_TO_CHOICE.index(example["resourceType"])

# Async function to get model response
async def get_model_response(example, client, semaphore):
    async with semaphore:
        prompt = doc_to_text(example)
        
        try:
            completion = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=20
            )
            response = completion.choices[0].message.content.strip()
            
            # Try to match the response to one of the valid choices
            for choice in DOC_TO_CHOICE:
                if choice.lower() in response.lower():
                    return choice
            
            # If no match found, return the raw response
            return response
        except Exception as e:
            logger.error(f"Error processing example: {e}")
            return None

# Main evaluation function
async def evaluate_fhir_generation_async(dataset_path=DATASET_PATH, test_split=TEST_SPLIT, batch_size=None, concurrent=MAX_CONCURRENT):
    # Set up Hugging Face
    api_token = "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"
    huggingface_hub.login(api_token)
    
    # Initialize OpenAI client
    # client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    # client = AsyncOpenAI(
    # api_key="KEY",
    # base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    # )
    client = AsyncOpenAI(api_key="KEY", base_url="https://api.deepseek.com")
    
    logger.info("Loading dataset...")
    dataset = load_dataset(dataset_path, split=test_split)
    
    # Limit dataset size if batch_size is specified
    if batch_size is not None:
        dataset = dataset.select(range(min(batch_size, len(dataset))))
    
    # Create a semaphore to limit concurrent API calls
    semaphore = asyncio.Semaphore(concurrent)
    
    logger.info(f"Starting evaluation on {len(dataset)} examples with {concurrent} concurrent requests...")
    tasks = [get_model_response(example, client, semaphore) for example in dataset]
    
    # Process all examples in parallel with progress bar
    predictions = await tqdm_asyncio.gather(*tasks, desc="Evaluating")
    
    # Collect results
    y_true = []
    y_pred = []
    correct_count = 0
    total_valid = 0
    
    for i, (example, prediction) in enumerate(zip(dataset, predictions)):
        if prediction is None:
            continue
            
        total_valid += 1
        gold_index = doc_to_target(example)
        gold_label = DOC_TO_CHOICE[gold_index]
        
        try:
            pred_index = DOC_TO_CHOICE.index(prediction) if prediction in DOC_TO_CHOICE else -1
            if pred_index >= 0:
                y_true.append(gold_index)
                y_pred.append(pred_index)
                if prediction == gold_label:
                    correct_count += 1
        except ValueError:
            logger.warning(f"Prediction '{prediction}' not in valid choices")
    
    # Calculate accuracy and F1 score
    accuracy = correct_count / total_valid if total_valid > 0 else 0.0
    
    # Calculate micro F1 score
    micro_f1 = f1_score(y_true, y_pred, average='micro') if y_true else 0.0
    
    logger.info(f"Number of valid examples: {total_valid}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Micro F1 Score: {micro_f1:.4f}")
    
    return {
        "accuracy": accuracy,
        "micro_f1": micro_f1,
        "total_examples": total_valid
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate FHIR Resource Identification")
    parser.add_argument("--dataset", type=str, default=DATASET_PATH, help="Dataset path")
    parser.add_argument("--split", type=str, default=TEST_SPLIT, help="Dataset split")
    parser.add_argument("--batch-size", type=int, default=None, help="Limit evaluation to N examples")
    parser.add_argument("--concurrent", type=int, default=MAX_CONCURRENT, help="Max concurrent requests")
    args = parser.parse_args()
    
    # Run the async evaluation
    asyncio.run(evaluate_fhir_generation_async(
        dataset_path=args.dataset,
        test_split=args.split,
        batch_size=args.batch_size,
        concurrent=args.concurrent
    ))