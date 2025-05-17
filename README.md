# Model Fine-Tuning Toolkit
This toolkit provides scripts, examples, and utilities to facilitate the process of adapting LLMs to handle classification task. It simplifies the workflow for fine-tuning on custom datasets.

It handles data loading, formatting for OpenAI's requirements, uploading datasets, creating and monitoring fine-tuning jobs, and evaluating the performance of both the base and fine-tuned models.

## Overview

The script automates several key steps in the OpenAI fine-tuning process:
* Loading training and testing datasets.
* Transforming data into the JSONL format required by OpenAI for chat model fine-tuning.
* Uploading the formatted training file to OpenAI.
* Initiating a fine-tuning job with specified models and hyperparameters.
* Polling the status of the fine-tuning job until completion.
* Evaluating the classification accuracy of the base model and the newly fine-tuned model.

## Features

* **Data Handling**: Loads data from TSV files and prepares it for fine-tuning.
* **Generalized Prompts**: Uses a flexible prompt structure for classification that can be customized.
* **OpenAI API Integration**: Interacts with the OpenAI API for file uploads, fine-tuning jobs, and completions.
* **Automated Polling**: Monitors fine-tuning jobs and retrieves the fine-tuned model ID upon success.
* **Evaluation**: Calculates and displays classification accuracy for test data.

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/JasonZhangHub/finetuning_toolkit.git
cd finetuning_toolkit
```

### 2. Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

```bash
# Create a virtual environment (e.g., named .venv)
python -m venv .venv

# Activate the virtual environment
# On macOS and Linux:
source .venv/bin/activate
# On Windows:
.\.venv\Scripts\activate
```

### 3. Install Dependencies

This script relies on `openai`, `pydantic`, and `python-dotenv`. If you have a `pyproject.toml` file (as described in a previous query) for Poetry:

Then, install dependencies using Poetry:

```bash
poetry install
```

Alternatively, if you are not using Poetry, you can create a `requirements.txt` file.
```
# requirements.txt
python-dotenv==1.1.0
openai==1.79.0
google-genai==1.15.0
pydantic==2.11.4
```

And install using pip:

```bash
pip install -r requirements.txt
```

### 4. OpenAI API Key

Create a file named `.env` in the root of project directory and add the key.

## Data Preparation
- Format: The script expects training and testing data in `csv` or `tsv` files.
- Location: Place your data files in a subdirectory named `data/` relative to the script.
  - Training data: data/train.tsv
  - Test data: data/test.tsv
- Columns:
    - The script expects a text column (default name: text) containing the input samples to be classified.
    - It also expects a label column (default name: label) containing the true category for each text sample.
    - These default names can be changed in the main() function of the script (see TEXT_COLUMN and LABEL_COLUMN variables).

The script will automatically format the training data into the JSONL format required by OpenAI and save it as classification_training_data.jsonl (configurable via OUTPUT_JSONL_FILE)

## Configurations
Several parameters can be configured directly within the `main()` function of the script:

- `TRAIN_FILE`, `TEST_FILE`: Paths to your data files.
- `TEXT_COLUMN`, `LABEL_COLUMN`: Names of the relevant columns in your data files.
- `OUTPUT_JSONL_FILE`: Name of the formatted JSONL file for OpenAI.
- `BASE_MODEL_FOR_FINETUNING`: The OpenAI base model ID you wish to fine-tune (e.g., 'gpt-4o-mini-2024-07-18', 'gpt-3.5-turbo').
- **Prompt Customization**:
    - `PROMPT_INSTRUCTION`
    - `CATEGORY_LIST_TAG`
    - `ITEM_WRAPPER_TAG`
    - `OUTPUT_WRAPPER_TAG`
  These allow you to change the structure and wording of the prompts used for fine-tuning and inference.

- **Fine-Tuning Hyperparameters** (in start_fine_tuning_job call within main()):

    - `n_epochs`: Number of training epochs.

    - `batch_size`, `learning_rate_multiplier` (can be set to "`auto`" for OpenAI defaults or specified).

    - `suffix`: An optional suffix for your fine-tuned model's name.

- **Polling Behavior** (in poll_fine_tuning_job call):
    - `poll_interval_seconds`, `timeout_seconds`

## Execute the Script

Ensure your virtual environment is activated and all dependencies are installed. Then, run the script from your terminal:
```bash
python openai_finetuner.py
```

## Customizing Prompts

The script uses a generalized prompt structure for the classification task. You can customize this by modifying these variables in the main() function:
- `PROMPT_INSTRUCTION`: The main instruction given to the model.
- `CATEGORY_LIST_TAG`: The XML-like tag used to wrap the list of possible categories in the prompt.
- `ITEM_WRAPPER_TAG`: The XML-like tag used to wrap the actual text item to be classified.
- `OUTPUT_WRAPPER_TAG`: The XML-like tag the model is instructed to use for its outputted category.

These parameters are used by the `_create_classification_prompt` helper function.


## Important Considerations
- **Costs**: Fine-tuning OpenAI models and making API calls (for file uploads and completions) will incur costs on your OpenAI account. Be mindful of the size of your training data and the number of epochs.
- **Model Availability**: Ensure the base model you specify (e.g., gpt-4o-mini-2024-07-18) is available for fine-tuning in your OpenAI account and region.
- **Data Quality**: The quality and quantity of your training data are crucial for the performance of the fine-tuned model. OpenAI provides guidelines on preparing your dataset.
- **Time**: Fine-tuning can take time, from several minutes to hours, depending on the dataset size, model, and OpenAI's current server load. The polling mechanism will keep you updated.
- **API Rate Limits**: Be aware of any API rate limits on your OpenAI account.