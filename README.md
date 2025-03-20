![FHIR-Workbench Github Banner](./images/FHIR-WORKBENCH-1.png)

# FHIR-Workbench

A comprehensive evaluation suite for testing Large Language Models (LLMs) on Fast Healthcare Interoperable Resources (FHIR) knowledge and capabilities.

## Overview

FHIR-Workbench provides standardized benchmarks to evaluate LLM performance on healthcare interoperability tasks. This repository contains implementations for four key FHIR tasks:

1. **FHIR-QA**: Tests general knowledge of FHIR concepts and standards through multiple-choice questions
2. **FHIR-RESTQA**: Evaluates specific understanding of FHIR RESTful API operations, queries, and interactions through multiple-choice questions
3. **FHIR-ResourceID**: Tests the ability to identify FHIR resource types based on their JSON structure and content
4. **Note2FHIR**: Assesses the ability to generate structured FHIR resources from patient clinical notes

## Dataset Access

All evaluation datasets are available on Hugging Face:
[FHIR-Workbench Collection](https://huggingface.co/collections/ikim-uk-essen/fhir-workbench-67daa05d2e7d1f15f6c0b145)

## Evaluation Methods

### Open-Source Models

FHIR tasks are integrated with [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) for standardized evaluation of open-source models.

#### Setup and Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/FHIR-Workbench.git
cd FHIR-Workbench

# Install lm-eval-harness with FHIR tasks
cd lm-evaluation-harness
pip install -e .
```

#### Running Evaluations

To evaluate an open-source model on all FHIR tasks:

```bash
lm_eval --model hf \
  --model_args pretrained=microsoft/phi-4 \
  --include_path lm_eval/tasks/fhir/ \
  --tasks fhir_qna,fhir_api,fhir_resource,fhir_generation \
  --output output \
  --log_samples \
  --apply_chat_template \
  --trust_remote_code
```

You can customize the evaluation by:
- Changing the model (`--model_args pretrained=...`)
- Selecting specific tasks (`--tasks ...`)
- Adjusting other parameters as needed

### Commercial Models

For proprietary models (OpenAI GPT, Google Gemini), use the provided scripts:

1. **run_qna_proprietary.py**: For FHIR-QA and FHIR-RESTQA tasks
2. **run_resource_proprietary.py**: For FHIR-ResourceID task
3. **run_generation_proprietary.py**: For Note2FHIR task

#### Setup

Set your API key as an environment variable:
```bash
# For OpenAI models
export OPENAI_API_KEY=your_api_key

# For other providers, edit the script to use your API key and endpoint
```

#### Running Evaluations

```bash
# For FHIR-QA task
python run_qna_proprietary.py --dataset ikim-uk-essen/FHIR-QA

# For FHIR-RESTQA task
python run_qna_proprietary.py --dataset ikim-uk-essen/FHIR-RESTQA

# For FHIR-ResourceID task
python run_resource_proprietary.py

# For Note2FHIR task
python run_generation_proprietary.py
```

Each script accepts additional parameters like `--batch-size` and `--concurrent` to control evaluation behavior.

## Leaderboard

We maintain a comprehensive leaderboard tracking performance of various LLMs on FHIR-specific tasks:
[FHIR-Workbench Leaderboard](https://ahmedidr.github.io/fhir-workbench/)

The leaderboard currently includes evaluations of 16 models ranging from open-source models (7B-671B parameters) to closed-source commercial models. Models are ranked based on their average performance across all four FHIR tasks.

### Current Top Performers

| Rank | Model          | Size   | FHIR-QA | FHIR-RESTQA | FHIR-ResourceID | Note2FHIR | Avg   |
|------|----------------|--------|---------|-------------|-----------------|-----------|-------|
| #1   | GPT-4o         | Closed | 94.0%   | 92.7%       | 99.9%           | 34.7%     | 80.3% |
| #2   | Gemini-2-Flash | Closed | 94.0%   | 90.0%       | 96.9%           | 34.0%     | 78.7% |
| #3   | Gemini-1.5-Pro | Closed | 93.3%   | 91.3%       | 93.7%           | 34.3%     | 78.2% |

### Submit Your Model

Have a FHIR-capable model you want to include in our leaderboard? Visit the [leaderboard page](https://ahmedidr.github.io/fhir-workbench/) and submit your HuggingFace model repository URL for evaluation.

## Citation

If you use FHIR-Workbench in your research, please cite our paper:
[Coming soon]

## License

[License information]

