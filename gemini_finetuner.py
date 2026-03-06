import json
import os
import time
from typing import List, Dict, Any, Tuple, Optional

import yaml
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
import pandas as pd
from dotenv import load_dotenv


def load_config(config_path="config.yml") -> Optional[Dict[str, Any]]:
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
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


def initialize_gemini_client() -> bool:
    """
    Loads GOOGLE_API_KEY from .env and configures the Gemini client.
    Returns True if successful, False otherwise.
    """
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found in .env file.")
        print("Please create a .env file with your GOOGLE_API_KEY or set the environment variable.")
        return False
    try:
        genai.configure(api_key=api_key)
        print("Google Generative AI client configured successfully.")
        return True
    except Exception as e:
        print(f"Error configuring Google Generative AI client: {e}")
        return False


def load_classification_data_from_file(
    data_filepath: str,
    text_column: str,
    label_column: str,
    sep: str = "\t"
) -> Optional[Tuple[pd.DataFrame, List[str]]]:
    """
    Loads data from a single TSV/CSV file. Used for train and test sets.
    Returns DataFrame and a list of unique categories.
    """
    try:
        print(
            f"Loading data from: {data_filepath} (text: '{text_column}', label: '{label_column}')")
        df = pd.read_csv(data_filepath, sep=sep, dtype={
                         text_column: str, label_column: str})

        if text_column not in df.columns or label_column not in df.columns:
            print(
                f"ERROR: Required columns '{text_column}' or '{label_column}' not found in {data_filepath}.")
            return None

        # Ensure columns are string type to avoid issues with mixed types or numbers being treated as such
        df[text_column] = df[text_column].astype(str)
        df[label_column] = df[label_column].astype(str)

        categories = sorted(list(df[label_column].unique()))
        print(
            f"Data loaded. Found {len(df)} examples. Categories: {categories if categories else 'None found'}")
        return df, categories
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {data_filepath}")
        return None
    except Exception as e:
        print(f"ERROR: Could not load data from {data_filepath}: {e}")
        return None


def format_data_for_gemini_fine_tuning(
    dataframe: pd.DataFrame,
    text_column: str,
    label_column: str
) -> List[Dict[str, str]]:
    """
    Converts a DataFrame into the list of dictionaries format for Gemini fine-tuning.
    Each dictionary: {"text_input": "...", "output": "..."}
    """
    formatted_examples = []
    for _, row in dataframe.iterrows():
        formatted_examples.append({
            "text_input": row[text_column],
            "output": row[label_column]
        })
    return formatted_examples


def save_data_to_jsonl(
    data_list: List[Dict[str, str]],
    jsonl_filepath: str
) -> bool:
    """Saves a list of dictionaries to a JSONL file."""
    try:
        print(f"Saving formatted data to JSONL file: {jsonl_filepath}")
        with open(jsonl_filepath, 'w', encoding='utf-8') as f:
            for item in data_list:
                f.write(json.dumps(item) + '\n')
        print(f"Data successfully saved to {jsonl_filepath}")
        return True
    except IOError as e:
        print(f"ERROR: Could not write to file {jsonl_filepath}: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while saving to JSONL: {e}")
        return False


def upload_file_to_gemini(
    jsonl_filepath: str,
    display_name: Optional[str] = None
) -> Optional[genai.types.File]:
    """Uploads the JSONL training data file to Google."""
    if not os.path.exists(jsonl_filepath):
        print(
            f"ERROR: Training data file not found at {jsonl_filepath} for upload.")
        return None
    if not display_name:
        display_name = os.path.basename(jsonl_filepath)
    try:
        print(
            f"Uploading training file '{jsonl_filepath}' to Google (Display name: '{display_name}')...")
        uploaded_file = genai.upload_file(
            path=jsonl_filepath, display_name=display_name)
        print(
            f"File uploaded successfully to Google. File Name (Resource): {uploaded_file.name}, URI: {uploaded_file.uri}")
        return uploaded_file
    except Exception as e:
        print(
            f"ERROR: Failed to upload file '{jsonl_filepath}' to Google: {e}")
        return None


def start_gemini_fine_tuning_job(
    user_defined_id: str,
    base_model_id: str,
    training_data_source: Any,
    hyperparams_config: Dict[str, Any],
    tuned_model_display_name: Optional[str] = None
) -> Optional[genai.types.tuned_model.TunedModelOperation]:
    """Launches a new fine-tuning job on Gemini."""
    print(
        f"\nAttempting to launch Gemini fine-tuning job for new model ID: '{user_defined_id}'")
    print(f"  Base model: {base_model_id}")
    if isinstance(training_data_source, genai.types.File):
        print(
            f"  Training data file: {training_data_source.name} ({training_data_source.uri})")
    else:
        print(
            f"  Training data: In-memory list of {len(training_data_source)} examples.")
    print(f"  Hyperparameters: {hyperparams_config}")

    if not tuned_model_display_name:
        tuned_model_display_name = f"{user_defined_id} (Tuned {base_model_id})"

    # Construct Hyperparameters object, filtering for expected keys by Gemini API
    valid_hyperparam_keys = {'epoch_count', 'batch_size', 'learning_rate'}
    actual_hyperparams = {k: v for k, v in hyperparams_config.items(
    ) if k in valid_hyperparam_keys and v is not None}

    gemini_hyperparams = None
    if actual_hyperparams:  # Only create if there are valid hyperparams
        gemini_hyperparams = genai.types.Hyperparameters(**actual_hyperparams)

    tuning_task = genai.types.TuningTask(
        training_data=training_data_source,
        hyperparameters=gemini_hyperparams  # Pass None if no specific hyperparams are set
    )

    try:
        operation = genai.create_tuned_model(
            id=user_defined_id,
            source_model=base_model_id,
            tuning_task=tuning_task,
            display_name=tuned_model_display_name
        )
        print(f"Fine-tuning job successfully submitted to Gemini.")
        print(f"  Operation Name: {operation.operation.name}")
        print(
            f"  Your tuned model, when ready, will be: tunedModels/{user_defined_id}")
        return operation
    except google_exceptions.AlreadyExists:
        print(
            f"ERROR: A tuned model with ID '{user_defined_id}' already exists.")
        print("Please choose a different 'user_defined_tuned_model_id' in config.yml or delete the existing one via Google AI Studio/API.")
        return None
    except Exception as e:
        print(f"ERROR: Failed to launch Gemini fine-tuning job: {e}")
        return None


def monitor_fine_tuning_progress(
    operation: genai.types.tuned_model.TunedModelOperation,
    poll_interval: int,
    timeout_seconds: int
) -> Optional[str]:
    """Monitors Gemini fine-tuning progress."""
    print(
        f"\nMonitoring fine-tuning progress for operation: {operation.operation.name}")
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        if operation.done():
            print("Fine-tuning operation reports as done.")
            try:
                result_tuned_model = operation.result()
                if result_tuned_model:
                    print(
                        f"  Operation result received. Tuned model name from result: {result_tuned_model.name}")
                    final_model_check = genai.get_tuned_model(
                        name=result_tuned_model.name)  # Refresh
                    print(
                        f"  Final check: Model '{final_model_check.name}' current state: {final_model_check.state}")
                    if final_model_check.state == genai.types.tuned_model.State.ACTIVE:
                        print(
                            f"SUCCESS: Fine-tuning complete. Model '{final_model_check.name}' is ACTIVE.")
                        return final_model_check.name
                    else:
                        print(
                            f"WARNING: Operation done, but model state is '{final_model_check.state}'. Expected ACTIVE.")
                        return None
                else:
                    print(
                        "WARNING: Operation done, but no tuned model object returned in result.")
                    return None
            except Exception as e:
                print(
                    f"ERROR: Fine-tuning operation completed with an error or failed to get result: {e}")
                if hasattr(operation, 'metadata') and operation.metadata:
                    print(
                        f"  Operation metadata at failure: {operation.metadata}")
                return None
        else:
            progress_message = "Polling..."
            # Example of accessing metadata fields if they exist (actual fields may vary)
            if hasattr(operation, 'metadata') and operation.metadata:
                if hasattr(operation.metadata, 'total_steps') and hasattr(operation.metadata, 'completed_steps'):
                    progress_message = f"Progress: {operation.metadata.completed_steps}/{operation.metadata.total_steps} steps"
                elif hasattr(operation.metadata, 'progress_percent'):
                    progress_message = f"Progress: {operation.metadata.progress_percent}%"
            elapsed_time = int(time.time() - start_time)
            print(
                f"  Status: In progress... {progress_message} (Elapsed: {elapsed_time}s)")
        time.sleep(poll_interval)

    print(
        f"TIMEOUT: Fine-tuning job did not complete within {timeout_seconds // 60} minutes.")
    return None


def list_gemini_tuned_models():
    """Lists fine-tuned models for the configured API key."""
    print("\n--- Listing Your Fine-Tuned Gemini Models ---")
    try:
        models = list(genai.list_tuned_models())
        if not models:
            print("No fine-tuned models found.")
            return
        print(f"Found {len(models)} fine-tuned model(s):")
        for model in models:
            print(
                f"  Display Name: {model.display_name}, ID: {model.id}, Full Name: {model.name}, State: {model.state}")
    except Exception as e:
        print(f"ERROR: Could not list fine-tuned models: {e}")


def classify_items_with_gemini_model(
    # Full name, e.g., "tunedModels/my-id" or "models/gemini-1.5-flash-latest"
    model_resource_name: str,
    items_to_classify: List[str],
    is_base_model_eval: bool = False,
    categories_for_base_prompt: Optional[List[str]] = None
) -> List[str]:
    """
    Classifies items using a Gemini model.
    For base model evaluation, a simple instruction prompt is used.
    For fine-tuned models, the input is typically just the text.
    """
    print(
        f"\nClassifying {len(items_to_classify)} items using Gemini model: {model_resource_name}")
    predictions = []
    try:
        model = genai.GenerativeModel(model_name=model_resource_name)
    except Exception as e:
        print(
            f"ERROR: Failed to load Gemini model '{model_resource_name}': {e}")
        return ["ERROR_LOADING_MODEL"] * len(items_to_classify)

    for i, text_input in enumerate(items_to_classify):
        prompt_to_send = text_input
        if is_base_model_eval:
            # Simple zero-shot classification prompt for base models
            categories_str = ", ".join(
                categories_for_base_prompt) if categories_for_base_prompt else "the relevant category"
            prompt_to_send = (
                f"Classify the following text into one of these categories: {categories_str}.\n"
                f"Output only the category name.\n\nText: \"{text_input}\"\n\nCategory:"
            )
        try:
            response = model.generate_content(prompt_to_send)
            predicted_label = response.text.strip()
            predictions.append(predicted_label)
            print(
                f"  Item {i+1}: '{text_input[:60]}...' -> Predicted: '{predicted_label}'")
        except Exception as e:
            print(
                f"ERROR: Failed to classify item '{text_input[:60]}...': {e}")
            predictions.append("CLASSIFICATION_FAILED")
    return predictions


def calculate_accuracy(predictions: List[str], actual_labels: List[str]) -> Optional[float]:
    """Calculates classification accuracy."""
    if len(predictions) != len(actual_labels):
        print("ERROR: Predictions and actual labels lists differ in length. Cannot calculate accuracy.")
        return None
    if not actual_labels:
        return 0.0
    correct_count = sum(1 for pred, actual in zip(
        predictions, actual_labels) if pred == actual)
    accuracy = (correct_count / len(actual_labels)) * 100
    print(
        f"\nAccuracy: {accuracy:.2f}% ({correct_count} correct out of {len(actual_labels)})")
    return accuracy

# --- Main Workflow ---


def main_gemini_workflow():
    if not CONFIG:
        print("FATAL: Script cannot run without a valid configuration. Ensure 'config.yml' exists and is correct. Exiting.")
        return

    print("--- Starting Gemini Fine-Tuning Workflow for Classification (YAML Config) ---")

    if not initialize_gemini_client():
        print("FATAL: Failed to initialize Gemini client. Exiting.")
        return

    # Get configurations
    cfg_file_paths = CONFIG.get('file_paths', {})
    cfg_data_cols = CONFIG.get('data_columns', {})
    cfg_gemini = CONFIG.get('gemini_settings', {})
    cfg_script_behavior = CONFIG.get('script_behavior', {})
    cfg_polling = cfg_script_behavior.get('polling', {})

    data_dir = cfg_file_paths.get('data_dir', 'data')
    train_filename = cfg_file_paths.get('train_filename', 'train.tsv')
    test_filename = cfg_file_paths.get('test_filename', 'test.tsv')
    output_jsonl_filename = cfg_file_paths.get(
        'output_jsonl_filename', 'gemini_classification_train_data.jsonl')

    text_col = cfg_data_cols.get('text_column', 'text')
    label_col = cfg_data_cols.get('label_column', 'label')

    base_model_for_tuning = cfg_gemini.get('base_model_for_tuning')
    user_defined_tuned_model_id = cfg_gemini.get('user_defined_tuned_model_id')
    tuned_model_display_name = cfg_gemini.get('tuned_model_display_name')
    hyperparams_config = cfg_gemini.get('hyperparameters', {})
    base_model_for_eval = cfg_gemini.get(
        'base_model_for_evaluation', base_model_for_tuning)

    should_upload = cfg_script_behavior.get('upload_training_file', True)
    should_start_job = cfg_script_behavior.get('start_fine_tuning_job', True)
    should_eval_base = cfg_script_behavior.get('evaluate_base_model', True)
    should_eval_tuned = cfg_script_behavior.get('evaluate_tuned_model', True)
    eval_sample_size = cfg_script_behavior.get('evaluation_sample_size', 20)
    poll_interval = cfg_polling.get('interval_seconds', 60)
    timeout_seconds = cfg_polling.get('timeout_seconds', 10800)

    manual_uploaded_file_name = cfg_script_behavior.get(
        'manual_uploaded_file_name')
    manual_fine_tuned_model_name = cfg_script_behavior.get(
        'manual_fine_tuned_model_name')

    if not base_model_for_tuning or not user_defined_tuned_model_id:
        print("FATAL: 'base_model_for_tuning' or 'user_defined_tuned_model_id' not set in config.yml. Exiting.")
        return

    train_file_full_path = os.path.join(data_dir, train_filename)
    test_file_full_path = os.path.join(data_dir, test_filename)
    output_jsonl_full_path = os.path.join(data_dir, output_jsonl_filename)

    # --- 1. Load and Prepare Training Data ---
    print("\n--- 1. Loading and Preparing Training Data ---")
    load_train_result = load_classification_data_from_file(
        train_file_full_path, text_col, label_col)
    if not load_train_result:
        print("FATAL: Failed to load training data. Exiting.")
        return
    training_df, train_categories = load_train_result
    if not train_categories:
        print("FATAL: No categories found in training data. This is needed for base model evaluation prompt. Exiting.")
        return

    # --- 2. Format Training Data for Gemini ---
    print("\n--- 2. Formatting Training Data for Gemini ---")
    gemini_training_data = format_data_for_gemini_fine_tuning(
        training_df, text_col, label_col)
    if not save_data_to_jsonl(gemini_training_data, output_jsonl_full_path):
        print(
            f"FATAL: Failed to save formatted training data to {output_jsonl_full_path}. Exiting.")
        return

    # --- 3. Upload Training File ---
    uploaded_train_file_object = None
    if manual_uploaded_file_name:
        print(
            f"\n--- Using manual uploaded file name from config: {manual_uploaded_file_name} ---")
        # We need a File object. If only name is given, we might not be able to construct it easily.
        # For now, assume if manual_uploaded_file_name is given, user has handled this.
        # Ideally, genai.get_file(name=manual_uploaded_file_name) would be used if available and needed.
        # For create_tuned_model, we pass the File object from upload_file or the list of dicts.
        # This example will proceed assuming if manual_uploaded_file_name is set, the user might skip upload and job creation.
        # Let's adjust to use a File object if a manual name is provided, by trying to fetch it.
        try:
            uploaded_train_file_object = genai.get_file(
                name=manual_uploaded_file_name)
            print(
                f"Successfully retrieved manually specified file: {uploaded_train_file_object.name}")
        except Exception as e:
            print(
                f"Warning: Could not retrieve manually specified file '{manual_uploaded_file_name}': {e}. Upload step might be necessary.")
            if should_upload:  # If we couldn't get it, but upload is allowed, proceed to upload.
                pass
            # If we couldn't get it, and upload is disabled, then we can't proceed with file-based tuning.
            else:
                print("ERROR: Manual file specified but could not be retrieved, and upload is disabled. Cannot proceed with file-based tuning.")
                # Option: Fallback to in-memory data if small enough, or abort.
                # For simplicity, we'll rely on the next step to handle if `training_data_source` is None.
                pass

    # Only upload if not manually provided and retrieved, and flag is true
    if not uploaded_train_file_object and should_upload:
        print("\n--- 3. Uploading Training File to Gemini ---")
        uploaded_train_file_object = upload_file_to_gemini(
            output_jsonl_full_path,
            display_name=f"training_data_{user_defined_tuned_model_id}"
        )

    # Determine training data source for the job
    training_data_source_for_job = None
    if uploaded_train_file_object:
        training_data_source_for_job = uploaded_train_file_object
    # If upload disabled and no manual file, try in-memory
    elif not should_upload and not manual_uploaded_file_name:
        print("Upload disabled and no manual file. Using in-memory data for fine-tuning (if dataset is small).")
        if len(gemini_training_data) > 20000:  # Example threshold
            print(
                f"WARNING: In-memory dataset has {len(gemini_training_data)} examples, which might be too large. File upload is recommended.")
        training_data_source_for_job = gemini_training_data

    if not training_data_source_for_job:
        print("FATAL: No training data source (file or in-memory) available for fine-tuning. Exiting.")
        return

    # --- 4. Start Fine-Tuning Job & Poll for Completion ---
    fine_tuned_model_full_resource_name = None
    if manual_fine_tuned_model_name:
        print(
            f"\n--- Using manual fine_tuned_model_name from config: {manual_fine_tuned_model_name} ---")
        # Validate if this model exists and is active
        try:
            model_check = genai.get_tuned_model(
                name=manual_fine_tuned_model_name)
            if model_check.state == genai.types.tuned_model.State.ACTIVE:
                fine_tuned_model_full_resource_name = model_check.name
                print(
                    f"Manual fine-tuned model '{fine_tuned_model_full_resource_name}' is ACTIVE.")
            else:
                print(
                    f"Warning: Manual fine-tuned model '{manual_fine_tuned_model_name}' exists but state is {model_check.state}. Evaluation might fail.")
                # Still set it for potential eval
                fine_tuned_model_full_resource_name = model_check.name
        except Exception as e:
            print(
                f"Warning: Could not retrieve manually specified fine-tuned model '{manual_fine_tuned_model_name}': {e}")

    if not fine_tuned_model_full_resource_name and should_start_job:
        print("\n--- 4. Starting Gemini Fine-Tuning Job & Polling ---")
        # Check if model already exists and is active to avoid re-running, if not manually specified
        expected_model_name = f"tunedModels/{user_defined_tuned_model_id}"
        try:
            existing_model = genai.get_tuned_model(name=expected_model_name)
            if existing_model.state == genai.types.tuned_model.State.ACTIVE:
                print(
                    f"Fine-tuned model '{expected_model_name}' already exists and is ACTIVE. Skipping tuning job.")
                fine_tuned_model_full_resource_name = existing_model.name
            else:
                print(
                    f"Model '{expected_model_name}' exists but state is {existing_model.state}. If re-tuning is desired, delete it first or choose a new ID.")
        except google_exceptions.NotFound:
            print(
                f"No existing active model found with ID '{user_defined_tuned_model_id}'. Proceeding to launch new tuning job.")
            # Launch job only if it doesn't exist or wasn't active
            tuning_operation = start_gemini_fine_tuning_job(
                user_defined_id=user_defined_tuned_model_id,
                base_model_id=base_model_for_tuning,
                training_data_source=training_data_source_for_job,
                hyperparams_config=hyperparams_config,
                tuned_model_display_name=tuned_model_display_name
            )
            if tuning_operation:
                fine_tuned_model_full_resource_name = monitor_fine_tuning_progress(
                    tuning_operation, poll_interval, timeout_seconds
                )
            else:
                print("Failed to start fine-tuning job.")
    elif not fine_tuned_model_full_resource_name:  # If job start was skipped by config
        print("\n--- Skipping Gemini Fine-Tuning Job (as per config) ---")

    if not fine_tuned_model_full_resource_name:
        print("Fine-tuning process did not result in an active model or was skipped. Evaluation of tuned model will be skipped.")
    else:
        print(
            f"\nActive fine-tuned model resource name: {fine_tuned_model_full_resource_name}")

    # --- 5. List Fine-Tuning Jobs (Optional Check) ---
    list_gemini_tuned_models()

    # --- 6. Load Test Data ---
    print("\n--- 6. Loading Test Data for Evaluation ---")
    load_test_result = load_classification_data_from_file(
        test_file_full_path, text_col, label_col)
    if not load_test_result:
        print("ERROR: Failed to load test data. Skipping evaluation.")
    else:
        test_df, test_categories = load_test_result
        # Use train_categories for prompting base model to ensure consistency
        # If test_categories differ, it's a data issue.

        sample_test_texts = test_df[text_col].tolist()[:eval_sample_size]
        sample_actual_labels = test_df[label_col].tolist()[:eval_sample_size]

        if not sample_test_texts:
            print("No sample test items to evaluate.")
        else:
            # 6a. Evaluate Base Model
            if should_eval_base:
                print(
                    f"\n--- Evaluating Base Gemini Model ({base_model_for_eval}) ---")
                base_model_responses = classify_items_with_gemini_model(
                    # e.g., "models/gemini-1.5-flash-latest"
                    model_resource_name=base_model_for_eval,
                    items_to_classify=sample_test_texts,
                    is_base_model_eval=True,
                    # Use categories from training data for prompt consistency
                    categories_for_base_prompt=train_categories
                )
                calculate_accuracy(base_model_responses, sample_actual_labels)
            else:
                print("\n--- Skipping Base Model Evaluation (as per config) ---")

            # 6b. Evaluate Fine-Tuned Model
            if should_eval_tuned and fine_tuned_model_full_resource_name:
                print(
                    f"\n--- Evaluating Fine-Tuned Gemini Model ({fine_tuned_model_full_resource_name}) ---")
                ft_model_responses = classify_items_with_gemini_model(
                    model_resource_name=fine_tuned_model_full_resource_name,
                    items_to_classify=sample_test_texts
                )
                calculate_accuracy(ft_model_responses, sample_actual_labels)
            elif should_eval_tuned:
                print(
                    "\n--- Skipping Fine-Tuned Model Evaluation: No active fine-tuned model ID available. ---")
            else:
                print("\n--- Skipping Fine-Tuned Model Evaluation (as per config) ---")

    print("\n--- Gemini Fine-Tuning Workflow Completed ---")


if __name__ == "__main__":
    if CONFIG is None:
        # Attempt to create a dummy config.yml if it doesn't exist
        if not os.path.exists("config.yml"):
            dummy_config_content = """
# Example config.yml for Gemini Fine-Tuner - PLEASE UPDATE WITH YOUR ACTUAL VALUES
# Ensure GOOGLE_API_KEY is in a .env file

file_paths:
  data_dir: "data"
  train_filename: "train.tsv"
  test_filename: "test.tsv"
  output_jsonl_filename: "gemini_classification_train_data.jsonl"

data_columns:
  text_column: "text"
  label_column: "label"

gemini_settings:
  base_model_for_tuning: "gemini-1.5-flash-001"
  user_defined_tuned_model_id: "my-gemini-classifier-001" # Change for each new model
  tuned_model_display_name: "My Custom Ticket Classifier (Gemini)"
  hyperparameters:
    epoch_count: 10
    batch_size: 4
    learning_rate: 0.001
  base_model_for_evaluation: "gemini-1.5-flash-latest"

script_behavior:
  upload_training_file: true
  start_fine_tuning_job: true
  evaluate_base_model: true
  evaluate_tuned_model: true
  polling:
    interval_seconds: 60
    timeout_seconds: 10800 # 3 hours
  evaluation_sample_size: 20
  # manual_uploaded_file_name: null # e.g., "files/xxxxxxxx"
  # manual_fine_tuned_model_name: null # e.g., "tunedModels/your-id"
"""
            try:
                with open("config.yml", "w", encoding='utf-8') as f:
                    f.write(dummy_config_content)
                print(
                    "Created a dummy 'config.yml'. Please review and update it, then re-run.")
                print("Ensure your GOOGLE_API_KEY is in a .env file.")
            except IOError:
                print(
                    "Could not create a dummy 'config.yml'. Please create it manually based on documentation.")
        else:
            print(
                "Failed to load configuration from 'config.yml'. Please ensure it's present and correct.")
    else:
        # Create dummy data files if they don't exist for demonstration
        cfg_file_paths = CONFIG.get('file_paths', {})
        data_dir = cfg_file_paths.get('data_dir', 'data')
        train_filename = cfg_file_paths.get('train_filename', 'train.tsv')
        test_filename = cfg_file_paths.get('test_filename', 'test.tsv')

        cfg_data_cols = CONFIG.get('data_columns', {})
        text_col_name = cfg_data_cols.get('text_column', "text")
        label_col_name = cfg_data_cols.get('label_column', "label")

        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)

        dummy_train_path = os.path.join(data_dir, train_filename)
        if not os.path.exists(dummy_train_path):
            dummy_train_data = {
                text_col_name: [
                    "Gemini is great for summarization tasks.", "This new phone has amazing camera features.",
                    "I need help with my account billing.", "The support agent was very helpful today.",
                    "Learning to code with Python is fun.", "The movie had a surprising plot twist."
                ] * 4,  # 24 examples
                label_col_name: [
                    "AI & Technology", "Product Review",
                    "Customer Support", "Customer Support",
                    "Education", "Entertainment"
                ] * 4
            }
            pd.DataFrame(dummy_train_data).to_csv(
                dummy_train_path, sep="\t", index=False)
            print(f"Created dummy training data at {dummy_train_path}")

        dummy_test_path = os.path.join(data_dir, test_filename)
        if not os.path.exists(dummy_test_path):
            dummy_test_data = {
                text_col_name: [
                    "The latest AI advancements are impressive.", "This laptop is too expensive for its specs.",
                    "My internet service is down again.", "The concert was an unforgettable experience."
                ],
                label_col_name: [
                    "AI & Technology", "Product Review",
                    "Customer Support", "Entertainment"
                ]
            }
            pd.DataFrame(dummy_test_data).to_csv(
                dummy_test_path, sep="\t", index=False)
            print(f"Created dummy test data at {dummy_test_path}")

        main_gemini_workflow()
