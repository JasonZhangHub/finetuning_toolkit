import json
import os
import time
from typing import List, Dict, Any, Tuple, Optional

import yaml
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv


def load_config(config_path="config.yml") -> Optional[Dict[str, Any]]:
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded successfully from {config_path}")
        return config
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at {config_path}")
        return None
    except yaml.YAMLError as e:
        print(
            f"ERROR: Could not parse YAML configuration from {config_path}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading configuration: {e}")
        return None


CONFIG = load_config()


def initialize_openai_client() -> Optional[OpenAI]:
    """
    Loads environment variables and initializes the OpenAI client.
    Prioritizes OPENAI_API_KEY from .env file.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    try:
        client = OpenAI(api_key=api_key)
        print("OpenAI client initialized successfully.")
        return client
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        return None


def load_classification_data(
    train_file_path: str,
    test_file_path: str,
    sep: str = "\t",
    text_column: str = "text",
    label_column: str = "label"
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str], List[str], List[str], List[str]]]:
    """
    Loads training and testing data from specified TSV/CSV files.
    Uses text_column and label_column from global CONFIG.
    """
    if not CONFIG:
        print("ERROR: Configuration not loaded. Cannot proceed with loading data.")
        return None

    # Get column names from config, with fallback to function defaults if somehow missing
    cfg_data_cols = CONFIG.get('data_columns', {})
    text_col = cfg_data_cols.get('text_column', text_column)
    label_col = cfg_data_cols.get('label_column', label_column)

    try:
        print(
            f"Loading training data from: {train_file_path} (text: '{text_col}', label: '{label_col}')")
        train_data = pd.read_csv(train_file_path, sep=sep)
        print(
            f"Loading test data from: {test_file_path} (text: '{text_col}', label: '{label_col}')")
        test_data = pd.read_csv(test_file_path, sep=sep)

        if text_col not in train_data.columns or label_col not in train_data.columns:
            print(
                f"ERROR: Required columns '{text_col}' or '{label_col}' not found in training data.")
            return None
        if text_col not in test_data.columns or label_col not in test_data.columns:
            print(
                f"ERROR: Required columns '{text_col}' or '{label_col}' not found in test data.")
            return None

        categories = sorted(
            train_data[label_col].astype(str).unique().tolist())
        print(f"Discovered categories: {categories}")

        train_texts = train_data[text_col].astype(str).tolist()
        train_labels = train_data[label_col].astype(str).tolist()
        test_texts = test_data[text_col].astype(str).tolist()
        test_labels = test_data[label_col].astype(str).tolist()

        return train_data, test_data, train_texts, train_labels, test_texts, test_labels, categories

    except FileNotFoundError as e:
        print(f"ERROR: Data file not found: {e.filename}")
        return None
    except Exception as e:
        print(f"ERROR: Could not load classification data: {e}")
        return None


def _create_classification_prompt(
    item_payload: str,
    categories_list: List[str]
    # Prompt customization parameters will be read from global CONFIG
) -> str:
    """
    Helper function to create a generalized prompt for classification.
    Reads prompt customization from global CONFIG.
    """
    if not CONFIG:
        # Fallback to some very basic defaults if config is missing, though this shouldn't happen
        prompt_config = {
            'instruction': "Classify:",
            'category_list_tag': "cats",
            'item_wrapper_tag': "item",
            'output_wrapper_tag': "out"
        }
        print("WARNING: Using fallback prompt defaults as global CONFIG not found.")
    else:
        prompt_config = CONFIG.get('prompt_customization', {})

    instruction = prompt_config.get(
        'instruction', "Classify the provided text into one of the following categories:")
    category_list_tag = prompt_config.get('category_list_tag', "categories")
    item_wrapper_tag = prompt_config.get('item_wrapper_tag', "text_input")
    output_wrapper_tag = prompt_config.get('output_wrapper_tag', "category")

    categories_str = '\n'.join(categories_list)
    return f"""{instruction}
                <{category_list_tag}>
                {categories_str}
                </{category_list_tag}>

                Here is the text to classify:
                <{item_wrapper_tag}>{item_payload}</{item_wrapper_tag}>

                Respond with the chosen {output_wrapper_tag} using the following format: <{output_wrapper_tag}>Chosen {output_wrapper_tag} Label</{output_wrapper_tag}>
                """


def format_data_for_openai_chat_completions(
    dataframe: pd.DataFrame,
    categories_list: List[str]
    # text_column, label_column, and prompt params will be read from global CONFIG
) -> Optional[List[Dict[str, List[Dict[str, str]]]]]:
    """
    Converts a DataFrame into the JSONL format expected by OpenAI for fine-tuning.
    Reads column names and prompt customization from global CONFIG.
    """
    if not CONFIG:
        print("ERROR: Configuration not loaded. Cannot format data.")
        return None

    cfg_data_cols = CONFIG.get('data_columns', {})
    text_column = cfg_data_cols.get('text_column', "text")  # Fallback
    label_column = cfg_data_cols.get('label_column', "label")  # Fallback

    prompt_config = CONFIG.get('prompt_customization', {})
    output_wrapper_tag = prompt_config.get(
        'output_wrapper_tag', "category")  # Fallback

    json_objs = []
    for _, example in dataframe.iterrows():
        user_msg_content = _create_classification_prompt(
            item_payload=str(example[text_column]),
            categories_list=categories_list
            # _create_classification_prompt internally uses CONFIG for other prompt parts
        )
        assistant_msg_content = f"<{output_wrapper_tag}>{str(example[label_column])}</{output_wrapper_tag}>"
        messages = [
            {"role": "user", "content": user_msg_content},
            {"role": "assistant", "content": assistant_msg_content}
        ]
        json_objs.append({"messages": messages})
    return json_objs


def save_to_jsonl(data: List[Dict[str, Any]], file_path: str) -> bool:
    """Saves the formatted data to a JSONL file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for obj in data:
                json.dump(obj, f)
                f.write('\n')
        print(f"Formatted data saved to: {file_path}")
        return True
    except IOError as e:
        print(f"Error writing to {file_path}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error saving to JSONL: {e}")
        return False


def upload_file_to_openai(client: OpenAI, file_path: str, purpose: str = "fine-tune") -> Optional[str]:
    """Uploads a file to OpenAI."""
    if not os.path.exists(file_path):
        print(f"ERROR: File not found for upload: {file_path}")
        return None
    try:
        print(
            f"Uploading file '{file_path}' to OpenAI for purpose '{purpose}'...")
        with open(file_path, "rb") as f:
            response = client.files.create(file=f, purpose=purpose)
        print(f"File uploaded successfully. File ID: {response.id}")
        return response.id
    except Exception as e:
        print(f"Error uploading file to OpenAI: {e}")
        return None


def start_fine_tuning_job(
    client: OpenAI,
    training_file_id: str
    # Model name, hyperparameters, and suffix will be read from global CONFIG
) -> Optional[Dict[str, Any]]:
    """
    Creates a fine-tuning job on OpenAI.
    Reads model settings and hyperparameters from global CONFIG.
    """
    if not CONFIG:
        print("ERROR: Configuration not loaded. Cannot start fine-tuning job.")
        return None

    openai_cfg = CONFIG.get('openai_settings', {})
    model_name = openai_cfg.get('base_model_for_finetuning')
    hyperparams_cfg = openai_cfg.get('hyperparameters', {})
    n_epochs = hyperparams_cfg.get('n_epochs', "auto")  # Default to auto
    batch_size = hyperparams_cfg.get('batch_size', "auto")
    learning_rate_multiplier = hyperparams_cfg.get(
        'learning_rate_multiplier', "auto")
    suffix = openai_cfg.get('model_suffix', None)

    if not model_name:
        print("ERROR: 'base_model_for_finetuning' not found in configuration.")
        return None

    print(
        f"Starting fine-tuning job for model '{model_name}' with training file ID '{training_file_id}'.")
    print(
        f"  Epochs: {n_epochs}, Batch Size: {batch_size}, LR Multiplier: {learning_rate_multiplier}")
    if suffix:
        print(f"  Model Suffix: {suffix}")
    print("Be mindful of the costs associated with fine-tuning.")

    # Prepare hyperparameters, filtering out any that are None or not meant for the API call directly
    # The OpenAI library handles "auto" as a string for some parameters.
    hyperparameters_payload = {}
    if n_epochs is not None:
        hyperparameters_payload['n_epochs'] = n_epochs
    if batch_size is not None:
        hyperparameters_payload['batch_size'] = batch_size
    if learning_rate_multiplier is not None:
        hyperparameters_payload['learning_rate_multiplier'] = learning_rate_multiplier

    # Ensure hyperparameters_payload is not empty if all are None,
    # though OpenAI API might default them. It's safer to pass them if specified.
    if not hyperparameters_payload:  # If all were None, pass an empty dict or specific defaults
        # OpenAI defaults will be used if hyperparameters key is not present or value is empty dict
        # For explicit "auto", we pass them.
        pass

    job_payload = {
        "training_file": training_file_id,
        "model": model_name,
    }
    if hyperparameters_payload:  # Only add if there are actual hyperparams to set
        job_payload["hyperparameters"] = hyperparameters_payload

    if suffix:
        job_payload["suffix"] = suffix

    try:
        job = client.fine_tuning.jobs.create(**job_payload)
        print(
            f"Fine-tuning job created successfully. Job ID: {job.id}, Status: {job.status}")
        return job.to_dict()  # Convert to dict for easier handling if needed
    except Exception as e:
        print(f"Error creating fine-tuning job: {e}")
        return None  # Changed from raise to return None for consistency


def poll_fine_tuning_job(
    client: OpenAI,
    job_id: str
    # Polling interval and timeout will be read from global CONFIG
) -> Optional[str]:
    """
    Polls the status of an OpenAI fine-tuning job.
    Reads polling settings from global CONFIG.
    """
    if not CONFIG:
        print("ERROR: Configuration not loaded. Using default polling settings.")
        poll_interval_seconds = 30
        timeout_seconds = 7200
    else:
        polling_cfg = CONFIG.get('script_behavior', {}).get('polling', {})
        poll_interval_seconds = polling_cfg.get('interval_seconds', 30)
        timeout_seconds = polling_cfg.get('timeout_seconds', 7200)

    start_time = time.time()
    print(
        f"Starting to poll fine-tuning job: {job_id} (Interval: {poll_interval_seconds}s, Timeout: {timeout_seconds}s)")
    while time.time() - start_time < timeout_seconds:
        try:
            job = client.fine_tuning.jobs.retrieve(job_id)
            status = job.status
            current_time_elapsed = int(time.time() - start_time)
            print(
                f"  Job ID: {job_id}, Status: {status}, Fine-tuned Model: {job.fine_tuned_model}, Time Elapsed: {current_time_elapsed}s")

            if status == 'succeeded':
                if job.fine_tuned_model:
                    print(
                        f"Fine-tuning job {job_id} succeeded. Fine-tuned model ID: {job.fine_tuned_model}")
                    return job.fine_tuned_model
                else:
                    print(
                        f"ERROR: Job {job_id} succeeded but no fine-tuned model ID was returned. Check OpenAI dashboard.")
                    return None
            elif status in ['failed', 'cancelled']:
                error_info = job.error.message if job.error and job.error.message else "No detailed error information provided."
                print(
                    f"Fine-tuning job {job_id} {status}. Error: {error_info}")
                return None
            elif status in ['validating_files', 'queued', 'running', 'pending']:
                # Job is still in progress
                pass
            else:
                print(
                    f"Unknown or unexpected job status encountered: {status} for job {job_id}. Stopping polling.")
                return None
            time.sleep(poll_interval_seconds)
        except Exception as e:
            print(
                f"Error retrieving job status for {job_id}: {e}. Retrying in {poll_interval_seconds}s...")
            time.sleep(poll_interval_seconds)

    print(f"Fine-tuning job {job_id} timed out after {timeout_seconds // 60} minutes. Last known status: {status if 'status' in locals() else 'unknown'}.")
    return None


def list_fine_tuning_jobs(client: OpenAI, limit: int = 20) -> List[Dict[str, Any]]:
    """Lists the fine-tuning jobs for your organization."""
    print(f"\n--- Listing up to {limit} Fine-Tuning Jobs ---")
    try:
        jobs_response = client.fine_tuning.jobs.list(limit=limit)
        job_list = [job.to_dict()
                    for job in jobs_response.data]  # Access .data for the list
        if not job_list:
            print("No fine-tuning jobs found.")
        else:
            print(f"Found {len(job_list)} jobs:")
            for job_info in job_list:
                status_str = f"  ID: {job_info.get('id')}, Model: {job_info.get('model')}, Status: {job_info.get('status')}"
                if job_info.get('fine_tuned_model'):
                    status_str += f", Fine-tuned ID: {job_info.get('fine_tuned_model')}"
                print(status_str)
        return job_list
    except Exception as e:
        print(f"Error listing fine-tuning jobs: {e}")
        return []


def classify_items_with_model(
    client: OpenAI,
    items_to_classify: List[str],
    model_id: str,
    categories_list: List[str]
    # Prompt customization, temperature, max_tokens will be read from global CONFIG
) -> List[str]:
    """
    Classifies a list of items using a specified OpenAI model.
    Reads prompt, temperature, and max_tokens from global CONFIG.
    """
    if not CONFIG:
        print("ERROR: Configuration not loaded. Using default classification settings.")
        temperature = 0.0
        max_tokens = 200
        output_wrapper_tag = "category"
    else:
        # Assuming these might be added to config.yml under openai_settings or a new section
        openai_cfg = CONFIG.get('openai_settings', {})
        temperature = openai_cfg.get('classification_temperature', 0.0)
        max_tokens = openai_cfg.get('classification_max_tokens', 200)
        prompt_cfg = CONFIG.get('prompt_customization', {})
        output_wrapper_tag = prompt_cfg.get('output_wrapper_tag', "category")

    responses = []
    stop_sequence = [f"</{output_wrapper_tag}>"]

    print(
        f"\nClassifying {len(items_to_classify)} items using model: {model_id}")
    for i, item_payload in enumerate(items_to_classify):
        user_prompt_content = _create_classification_prompt(
            item_payload=item_payload,
            categories_list=categories_list
            # _create_classification_prompt uses CONFIG for other prompt parts
        )
        try:
            completion = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": user_prompt_content}],
                temperature=temperature,
                stop=stop_sequence,
                max_tokens=max_tokens,
            )
            response_content = completion.choices[0].message.content.strip()

            predicted_label = response_content
            # Extract content within <output_wrapper_tag>...</output_wrapper_tag>
            # Model might stop before closing tag, or include it.
            if predicted_label.startswith(f"<{output_wrapper_tag}>"):
                predicted_label = predicted_label[len(
                    f"<{output_wrapper_tag}>"):].strip()
            # Defensive, if stop didn't catch it
            if predicted_label.endswith(f"</{output_wrapper_tag}>"):
                predicted_label = predicted_label[:-
                                                  len(f"</{output_wrapper_tag}>")].strip()

            responses.append(predicted_label)
            print(
                f"  Item {i+1}: '{item_payload[:50]}...' -> Predicted: '{predicted_label}'")
        except Exception as e:
            print(f"  Error classifying item '{item_payload[:50]}...': {e}")
            responses.append("CLASSIFICATION_ERROR")
    return responses


def calculate_accuracy(predicted_labels: List[str], actual_labels: List[str]) -> float:
    """Calculates the percentage of correctly classified predictions."""
    if not actual_labels:
        return 0.0
    if len(predicted_labels) != len(actual_labels):
        print("Warning: Predicted and actual labels lists differ in length. Accuracy might be misleading.")
        return 0.0

    num_correct = sum(p == a for p, a in zip(predicted_labels, actual_labels))
    accuracy = round(100 * num_correct / len(actual_labels), 2)
    print(
        f"Accuracy calculated: {accuracy}% ({num_correct} correct out of {len(actual_labels)})")
    return accuracy


def main():
    """
    Main function to execute the classification fine-tuning process.
    Reads all configurations from config.yml.
    """
    if not CONFIG:
        print("FATAL: Script cannot run without a valid configuration. Exiting.")
        return

    print("--- Starting OpenAI Fine-Tuning Workflow for Classification (YAML Config) ---")

    client = initialize_openai_client()
    if not client:
        print("FATAL: Failed to initialize OpenAI client. Exiting.")
        return

    # Get file paths from config
    cfg_file_paths = CONFIG.get('file_paths', {})
    data_dir = cfg_file_paths.get('data_dir', 'data')
    train_filename = cfg_file_paths.get('train_filename', 'train.tsv')
    test_filename = cfg_file_paths.get('test_filename', 'test.tsv')
    output_jsonl_filename = cfg_file_paths.get(
        'output_jsonl_filename', 'classification_training_data.jsonl')

    train_file_full_path = os.path.join(data_dir, train_filename)
    test_file_full_path = os.path.join(data_dir, test_filename)
    output_jsonl_full_path = os.path.join(
        data_dir, output_jsonl_filename)  # Store in data dir too

    # --- 1. Load and Prepare Data ---
    print("\n--- 1. Loading and Preparing Data ---")
    load_data_result = load_classification_data(
        train_file_full_path,
        test_file_full_path
        # text_column and label_column are now handled internally by load_classification_data using CONFIG
    )
    if not load_data_result:
        print("FATAL: Failed to load data. Exiting.")
        return

    training_df, test_df, _, _, test_texts, test_labels, categories = load_data_result

    print(f"\nTraining examples head:\n{training_df.head()}")
    if not categories:
        print("FATAL: No categories found in training data. Exiting.")
        return
    print(f"Categories for classification: {categories}")

    # --- 2. Format Data for OpenAI ---
    print("\n--- 2. Formatting Training Data for OpenAI ---")
    training_json_data = format_data_for_openai_chat_completions(
        training_df,
        categories
    )
    if not training_json_data:
        print("FATAL: Failed to format training data. Exiting.")
        return

    if not save_to_jsonl(training_json_data, output_jsonl_full_path):
        print(
            f"FATAL: Failed to save formatted data to {output_jsonl_full_path}. Exiting.")
        return

    # --- Script Behavior Flags ---
    script_behavior_cfg = CONFIG.get('script_behavior', {})
    should_upload_file = script_behavior_cfg.get('upload_training_file', True)
    should_start_job = script_behavior_cfg.get('start_fine_tuning_job', True)
    should_eval_base = script_behavior_cfg.get('evaluate_base_model', True)
    should_eval_tuned = script_behavior_cfg.get('evaluate_tuned_model', True)
    eval_sample_size = script_behavior_cfg.get('evaluation_sample_size', 20)

    # --- 3. Upload Training File ---
    training_file_id = None
    if should_upload_file:
        print("\n--- 3. Uploading Training File ---")
        training_file_id = upload_file_to_openai(
            client, output_jsonl_full_path)
        if not training_file_id:
            print(
                "ERROR: Failed to upload training file. Subsequent steps requiring it may fail.")
            # Decide if to stop or allow proceeding if user has a manual ID
            # For now, we'll let it proceed and it will fail at job creation if ID is needed and missing.
    else:
        print("\n--- Skipping Training File Upload (as per config) ---")
        # Allow user to manually set this in config.yml if needed for testing specific parts
        # e.g., add a field like: manual_training_file_id: "file-xxxxxxxx" under script_behavior
        manual_file_id = script_behavior_cfg.get('manual_training_file_id')
        if manual_file_id:
            training_file_id = manual_file_id
            print(
                f"Using manual training_file_id from config: {training_file_id}")

    # --- 4. Start Fine-Tuning Job & Poll for Completion ---
    fine_tuned_model_id = None
    if should_start_job:
        if training_file_id:
            print("\n--- 4. Starting Fine-Tuning Job & Polling ---")
            job_details_dict = start_fine_tuning_job(
                client,
                training_file_id
                # model, hyperparameters, suffix are read from CONFIG internally
            )
            if job_details_dict and job_details_dict.get('id'):
                job_id = job_details_dict['id']
                print(
                    f"Fine-tuning job submitted. Job ID: {job_id}. Polling for completion...")
                fine_tuned_model_id = poll_fine_tuning_job(client, job_id)
                if fine_tuned_model_id:
                    print(
                        f"Successfully obtained fine-tuned model ID: {fine_tuned_model_id}")
                else:
                    print("Could not obtain fine-tuned model ID after polling.")
            else:
                print("Failed to submit fine-tuning job or get Job ID.")
        else:
            print(
                "\n--- Skipping Fine-Tuning Job: Training file ID not available/not uploaded. ---")
    else:
        print("\n--- Skipping Fine-Tuning Job (as per config) ---")
        manual_ft_id = script_behavior_cfg.get('manual_fine_tuned_model_id')
        if manual_ft_id:
            fine_tuned_model_id = manual_ft_id
            print(
                f"Using manual fine_tuned_model_id from config: {fine_tuned_model_id}")

    # --- 5. List Fine-Tuning Jobs (Optional Check) ---
    list_fine_tuning_jobs(client)

    # --- 6. Evaluate Models ---
    print(f"\n--- 6. Evaluating Models (Sample Size: {eval_sample_size}) ---")
    # Ensure test_texts and test_labels are available and sliced
    if not test_texts or not test_labels:
        print("ERROR: Test data (texts or labels) not available for evaluation.")
    else:
        sample_test_items = test_texts[:eval_sample_size]
        sample_actual_labels = test_labels[:eval_sample_size]

        if not sample_test_items:
            print("No sample test items to evaluate.")
        else:
            # 6a. Evaluate Base Model
            if should_eval_base:
                base_model_to_eval = CONFIG.get('openai_settings', {}).get(
                    'base_model_for_finetuning', 'gpt-3.5-turbo')  # Fallback
                print(
                    f"\n--- Evaluating Base Model ({base_model_to_eval}) ---")
                base_model_responses = classify_items_with_model(
                    client,
                    items_to_classify=sample_test_items,
                    model_id=base_model_to_eval,
                    categories_list=categories
                )
                calculate_accuracy(base_model_responses, sample_actual_labels)
            else:
                print("\n--- Skipping Base Model Evaluation (as per config) ---")

            # 6b. Evaluate Fine-Tuned Model
            if should_eval_tuned:
                if fine_tuned_model_id:
                    print(
                        f"\n--- Evaluating Fine-Tuned Model ({fine_tuned_model_id}) ---")
                    ft_model_responses = classify_items_with_model(
                        client,
                        items_to_classify=sample_test_items,
                        model_id=fine_tuned_model_id,
                        categories_list=categories
                    )
                    calculate_accuracy(ft_model_responses,
                                       sample_actual_labels)
                else:
                    print(
                        "\n--- Skipping Fine-Tuned Model Evaluation: Fine-tuned model ID not available. ---")
            else:
                print("\n--- Skipping Fine-Tuned Model Evaluation (as per config) ---")

    print("\n--- OpenAI Fine-Tuning Workflow Completed ---")


if __name__ == "__main__":
    main()
