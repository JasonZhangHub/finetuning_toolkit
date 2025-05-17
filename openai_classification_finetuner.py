import json
import os
import time
from typing import List, Dict, Any, Tuple, Optional

from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv


def initialize_openai_client() -> OpenAI:
    """
    Loads environment variables from a .env file (if present)
    and initializes the OpenAI client.

    The OpenAI API key is expected to be in an environment variable
    named OPENAI_API_KEY or accessible through default OpenAI client discovery.

    Returns:
        OpenAI: An initialized OpenAI client instance.
    """
    load_dotenv()
    client = OpenAI()
    print("OpenAI client initialized.")
    return client


def load_classification_data(
        train_file_path: str,
        test_file_path: str,
        sep: str = "\t",
        text_column: str = "text",
        label_column: str = "label",
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str], List[str], List[str]]:
    """
    Loads training and testing data from specified TSV/CSV files.

    Args:
        train_file_path (str): Path to the training data file.
        test_file_path (str): Path to the test data file.
        sep (str): Separator for the data files (default is tab).
        text_column (str): Name of the column containing the input text.
        label_column (str): Name of the column containing the labels.

    Returns:
        Tuple: Contains:
            - training_df (pd.DataFrame): DataFrame for training.
            - test_df (pd.DataFrame): DataFrame for testing.
            - training_texts (List[str]): List of training texts.
            - training_labels (List[str]): List of training labels.
            - test_texts (List[str]): List of test texts.
            - test_labels (List[str]): List of test labels.
            - categories (List[str]): Sorted list of unique labels from the training set.
    """
    train_data = pd.read_csv(train_file_path, sep=sep)
    test_data = pd.read_csv(test_file_path, sep=sep)

    categories = sorted(train_data[label_column].unique().tolist())

    train_texts = train_data[text_column].tolist()
    train_labels = train_data[label_column].tolist()
    test_texts = test_data[text_column].tolist()
    test_labels = test_data[label_column].tolist()

    return train_data, test_data, train_texts, train_labels, test_texts, test_labels, categories


def _create_classification_prompt(
    item_payload: str,
    categories_list: List[str],
    prompt_instruction: str = "Classify the provided text into one of the following categories:",
    category_list_tag: str = "categories",
    item_wrapper_tag: str = "text_input",
    output_wrapper_tag: str = "category"
) -> str:
    """
    Helper function to create a generalized prompt for classification.

    Args:
        item_payload (str): The actual text content of the item to classify.
        categories_list (List[str]): List of possible category labels.
        prompt_instruction (str): The main instruction for the classification task.
        category_list_tag (str): XML-like tag to wrap the list of categories (e.g., "categories").
        item_wrapper_tag (str): XML-like tag to wrap the input item_payload (e.g., "ticket_text").
        output_wrapper_tag (str): XML-like tag expected for the output label (e.g., "category").

    Returns:
        str: The formatted prompt string.
    """
    categories_str = '\n'.join(categories_list)
    return f"""{prompt_instruction}
                <{category_list_tag}>
                {categories_str}
                </{category_list_tag}>

                Here is the text to classify:
                <{item_wrapper_tag}>{item_payload}</{item_wrapper_tag}>

                Respond with the chosen {output_wrapper_tag} using the following format: <{output_wrapper_tag}>Chosen {output_wrapper_tag} Label</{output_wrapper_tag}>
                """


def format_data_for_openai_chat_completions(
    dataframe: pd.DataFrame,
    text_column: str,
    label_column: str,
    categories_list: List[str],
    # Parameters for generic prompt creation
    prompt_instruction: str = "Classify the provided text into one of the following categories:",
    category_list_tag: str = "categories",
    item_wrapper_tag: str = "text_input",
    output_wrapper_tag: str = "category"
) -> List[Dict[str, List[Dict[str, str]]]]:
    """
    Converts a DataFrame into the JSONL format expected by OpenAI for fine-tuning
    chat completion models, using a generalized prompt structure.

    Args:
        dataframe (pd.DataFrame): DataFrame containing the text and label columns.
        text_column (str): Name of the column with input text.
        label_column (str): Name of the column with labels.
        categories_list (List[str]): List of possible categories for the prompt.
        prompt_instruction (str): Main instruction for the classification task.
        category_list_tag (str): Tag for wrapping the list of categories.
        item_wrapper_tag (str): Tag for wrapping the input item.
        output_wrapper_tag (str): Tag for the expected output label.


    Returns:
        List[Dict[str, List[Dict[str, str]]]]: A list of message objects.
    """
    json_objs = []
    for _, example in dataframe.iterrows():
        user_msg_content = _create_classification_prompt(
            item_payload=example[text_column],
            categories_list=categories_list,
            prompt_instruction=prompt_instruction,
            category_list_tag=category_list_tag,
            item_wrapper_tag=item_wrapper_tag,
            output_wrapper_tag=output_wrapper_tag
        )
        # Assistant's response should be just the chosen label wrapped in the output_wrapper_tag
        assistant_msg_content = f"<{output_wrapper_tag}>{example[label_column]}</{output_wrapper_tag}>"
        messages = [
            {"role": "user", "content": user_msg_content},
            {"role": "assistant", "content": assistant_msg_content}
        ]
        json_objs.append({"messages": messages})
    return json_objs


def save_to_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """
    Saves the formatted data to a JSONL file.

    Args:
        data (List[Dict[str, Any]]): List of JSON objects to save.
        file_path (str): Path to the output JSONL file.
    """
    with open(file_path, 'w') as f:
        for obj in data:
            json.dump(obj, f)
            f.write('\n')
    print(f"Formatted data saved to: {file_path}")


def upload_file_to_openai(client: OpenAI, file_path: str, purpose: str = "fine-tune") -> str:
    """
    Uploads a file to OpenAI.

    Args:
        client (OpenAI): The initialized OpenAI client.
        file_path (str): Path to the file to be uploaded.
        purpose (str): Purpose of the file (e.g., "fine-tune").

    Returns:
        str: The ID of the uploaded file.
    """
    print(f"Uploading file '{file_path}' to OpenAI for purpose '{purpose}'...")
    with open(file_path, "rb") as f:
        response = client.files.create(file=f, purpose=purpose)
    print(f"File uploaded successfully. File ID: {response.id}")
    return response.id


def start_fine_tuning_job(
    client: OpenAI,
    training_file_id: str,
    model_name: str,  # e.g., 'gpt-4o-mini-2024-07-18'
    batch_size: Optional[Any] = "auto",
    learning_rate_multiplier: Optional[Any] = "auto",
    n_epochs: Optional[Any] = "auto",
    suffix: Optional[str] = None
) -> Dict[str, Any]:
    """
    Creates a fine-tuning job on OpenAI.

    Args:
        client (OpenAI): The initialized OpenAI client.
        training_file_id (str): The ID of the uploaded training file.
        model_name (str): The base model to fine-tune (e.g., 'gpt-4o-mini-2024-07-18').
        batch_size (Optional[Any]): Batch size. "auto" or an integer.
        learning_rate_multiplier (Optional[Any]): Learning rate multiplier. "auto" or a float.
        n_epochs (Optional[Any]): Number of training epochs. "auto" or an integer.
        suffix (Optional[str]): A string of up to 18 characters that will be added to your fine-tuned model name.

    Returns:
        Dict[str, Any]: The response object from the OpenAI API representing the created job.
    """
    print(
        f"Starting fine-tuning job for model '{model_name}' with training file ID '{training_file_id}'.")
    print("Be mindful of the costs associated with fine-tuning.")

    hyperparameters = {
        "batch_size": batch_size,
        "learning_rate_multiplier": learning_rate_multiplier,
        "n_epochs": n_epochs,
    }
    # Filter out None values to use OpenAI defaults if not specified
    filtered_hyperparameters = {k: v for k,
                                v in hyperparameters.items() if v is not None}

    job_payload = {
        "training_file": training_file_id,
        "model": model_name,
    }
    # The new fine_tuning.jobs.create API uses a 'hyperparameters' dictionary
    # For older models or different API versions, this structure might vary.
    job_payload["hyperparameters"] = filtered_hyperparameters

    if suffix:
        job_payload["suffix"] = suffix

    try:
        job = client.fine_tuning.jobs.create(**job_payload)
        print(
            f"Fine-tuning job created successfully. Job ID: {job.id}, Status: {job.status}")
        return job.to_dict()
    except Exception as e:
        print(f"Error creating fine-tuning job: {e}")
        raise


def poll_fine_tuning_job(
    client: OpenAI,
    job_id: str,
    poll_interval_seconds: int = 30,
    timeout_seconds: int = 7200  # 2 hours, adjust as needed
) -> Optional[str]:
    """
    Polls the status of an OpenAI fine-tuning job until it completes, fails, or times out.

    Args:
        client (OpenAI): The initialized OpenAI client.
        job_id (str): The ID of the fine-tuning job to monitor.
        poll_interval_seconds (int): How often (in seconds) to check the job status.
        timeout_seconds (int): Maximum time (in seconds) to wait for the job to complete.

    Returns:
        Optional[str]: The fine-tuned model ID if the job succeeds, otherwise None.
    """
    start_time = time.time()
    print(f"Starting to poll fine-tuning job: {job_id}")
    while time.time() - start_time < timeout_seconds:
        try:
            job = client.fine_tuning.jobs.retrieve(job_id)
            status = job.status
            current_time_elapsed = int(time.time() - start_time)
            print(
                f"Job ID: {job_id}, Status: {status}, Fine-tuned Model: {job.fine_tuned_model}, Time Elapsed: {current_time_elapsed}s")

            if status == 'succeeded':
                if job.fine_tuned_model:
                    print(
                        f"Fine-tuning job {job_id} succeeded. Fine-tuned model ID: {job.fine_tuned_model}")
                    return job.fine_tuned_model
                else:
                    print(
                        f"Error: Job {job_id} succeeded but no fine-tuned model ID was returned by the API. Please check the OpenAI dashboard.")
                    return None  # Should ideally not happen if status is succeeded
            elif status in ['failed', 'cancelled']:
                error_info = job.error if job.error else "No detailed error information provided."
                print(
                    f"Fine-tuning job {job_id} {status}. Error: {error_info}")
                return None
            elif status in ['validating_files', 'queued', 'running']:
                # Job is still in progress, wait for the next poll interval
                pass
            else:
                print(
                    f"Unknown or unexpected job status encountered: {status} for job {job_id}. Stopping polling.")
                return None

            time.sleep(poll_interval_seconds)

        except Exception as e:
            print(
                f"Error retrieving job status for {job_id}: {e}. Retrying in {poll_interval_seconds}s...")
            # Wait before retrying on API error
            time.sleep(poll_interval_seconds)

    print(f"Fine-tuning job {job_id} timed out after {timeout_seconds} seconds of polling. Last known status: {job.status if 'job' in locals() else 'unknown'}.")
    return None


def list_fine_tuning_jobs(client: OpenAI,
                          limit: int = 20
                          ) -> List[Dict[str, Any]]:
    """
    Lists the fine-tuning jobs for your organization.

    Args:
        client (OpenAI): The initialized OpenAI client.
        limit (int): The maximum number of fine-tuning jobs to retrieve.

    Returns:
        List[Dict[str, Any]]: A list of fine-tuning job objects.
    """
    print(f"Listing up to {limit} fine-tuning jobs...")
    jobs = client.fine_tuning.jobs.list(limit=limit)
    job_list = [job.to_dict() for job in jobs.data]
    print(f"Found {len(job_list)} jobs.")
    return job_list


def classify_items_with_model(
    client: OpenAI,
    items_to_classify: List[str],
    model_id: str,
    categories_list: List[str],
    prompt_instruction: str = "Classify the provided text into one of the following categories:",
    category_list_tag: str = "categories",
    item_wrapper_tag: str = "text_input",
    output_wrapper_tag: str = "category",
    temperature: float = 0.0,
    max_tokens: int = 200
) -> List[str]:
    """
    Classifies a list of items using a specified OpenAI model and a generalized prompt.

    Args:
        client (OpenAI): The initialized OpenAI client.
        items_to_classify (List[str]): A list of item texts to classify.
        model_id (str): The ID of the OpenAI model to use.
        categories_list (List[str]): List of possible categories for the prompt.
        prompt_instruction (str): Main instruction for the classification task.
        category_list_tag (str): Tag for wrapping the list of categories.
        item_wrapper_tag (str): Tag for wrapping the input item.
        output_wrapper_tag (str): Tag for the expected output label.
        temperature (float): Sampling temperature.
        max_tokens (int): Maximum number of tokens for the completion.

    Returns:
        List[str]: A list of predicted category labels.
    """
    responses = []
    # The stop sequence should be the closing tag of the expected output.
    # The API will stop generating *before* this sequence.
    stop_sequence = [f"</{output_wrapper_tag}>"]

    print(
        f"Classifying {len(items_to_classify)} items using model: {model_id}")
    for i, item_payload in enumerate(items_to_classify):
        user_prompt_content = _create_classification_prompt(
            item_payload=item_payload,
            categories_list=categories_list,
            prompt_instruction=prompt_instruction,
            category_list_tag=category_list_tag,
            item_wrapper_tag=item_wrapper_tag,
            output_wrapper_tag=output_wrapper_tag
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

            # Expected response content (due to stop sequence) is like: "<output_wrapper_tag>ActualLabel"
            predicted_label = response_content
            if predicted_label.startswith(f"<{output_wrapper_tag}>"):
                predicted_label = predicted_label[len(
                    f"<{output_wrapper_tag}>"):].strip()
            # Defensive check
            if predicted_label.endswith(f"</{output_wrapper_tag}>"):
                predicted_label = predicted_label[:-
                                                  len(f"</{output_wrapper_tag}>")].strip()

            responses.append(predicted_label)
            print(
                f"Item {i+1}/{len(items_to_classify)} classified as: {predicted_label}")
        except Exception as e:
            print(
                f"Error classifying item {i+1} ('{item_payload[:50]}...'): {e}")
            responses.append("CLASSIFICATION_ERROR")  # Placeholder for errors
    return responses


def calculate_accuracy(predicted_labels: List[str], actual_labels: List[str]) -> float:
    """
    Calculates the percentage of correctly classified predictions.

    Args:
        predicted_labels (List[str]): List of labels predicted by the model.
        actual_labels (List[str]): List of true labels.

    Returns:
        float: Accuracy percentage, rounded to two decimal places.
    """
    if len(predicted_labels) != len(actual_labels):
        raise ValueError(
            "Predicted and actual labels lists must have the same length.")
    if not actual_labels:
        return 0.0

    num_correct = sum(p == a for p, a in zip(predicted_labels, actual_labels))
    accuracy = round(100 * num_correct / len(actual_labels), 2)
    print(
        f"Accuracy calculated: {accuracy}% ({num_correct}/{len(actual_labels)})")
    return accuracy


def main():
    """
    Main function to execute the classification fine-tuning process.
    It includes loading data, formatting it for OpenAI, uploading files,
    starting fine-tuning jobs, and evaluating models.
    """
    # --- Configuration ---
    TRAIN_FILE = 'data/train.tsv'
    TEST_FILE = 'data/test.tsv'
    TEXT_COLUMN = 'text'
    LABEL_COLUMN = 'label'
    OUTPUT_JSONL_FILE = 'classification_training_data.jsonl'

    BASE_MODEL_FOR_FINETUNING = 'gpt-4o-mini-2024-07-18'

    PROMPT_INSTRUCTION = "Please classify the following customer support ticket into one of these categories:"
    CATEGORY_LIST_TAG = "support_categories"
    ITEM_WRAPPER_TAG = "ticket_text"
    OUTPUT_WRAPPER_TAG = "assigned_category"

    # --- Initialize ---
    client = initialize_openai_client()

    # --- 1. Load and Prepare Data ---
    (training_df,
     test_df,
     training_texts,
     training_labels,
     test_texts,
     test_labels,
     categories
     ) = load_classification_data(TRAIN_FILE,
                                  TEST_FILE,
                                  text_column=TEXT_COLUMN,
                                  label_column=LABEL_COLUMN)

    print(f"\nTraining examples head:\n{training_df.head()}")
    print(f"Categories found: {categories}")

    # --- 2. Format Data for OpenAI ---
    training_json_data = format_data_for_openai_chat_completions(
        training_df, TEXT_COLUMN, LABEL_COLUMN, categories,
        prompt_instruction=PROMPT_INSTRUCTION,
        category_list_tag=CATEGORY_LIST_TAG,
        item_wrapper_tag=ITEM_WRAPPER_TAG,
        output_wrapper_tag=OUTPUT_WRAPPER_TAG
    )
    save_to_jsonl(training_json_data, OUTPUT_JSONL_FILE)

    # --- 3. Upload Training File ---
    training_file_id = None
    print("\n--- Uploading Training File ---")
    try:
        training_file_id = upload_file_to_openai(client, OUTPUT_JSONL_FILE)
        print(f"Training File ID for fine-tuning: {training_file_id}")
    except Exception as e:
        print(f"Failed to upload training file: {e}")
        training_file_id = None  # Ensure it's None if upload fails
    # # Example for testing without actual upload:
    # # training_file_id = "file-xxxxxxxxxxxxxxxxxxxxx" # Replace with your actual file ID if already uploaded

    # --- 4. Start Fine-Tuning Job & Poll for Completion (THIS WILL INCUR COSTS) ---
    fine_tuned_model_id = None
    if training_file_id:
        print("\n--- Starting Fine-Tuning Job ---")
        try:
            job_response = start_fine_tuning_job(
                client,
                training_file_id=training_file_id,
                model_name=BASE_MODEL_FOR_FINETUNING,
                n_epochs=3,  # "auto" or specific number
                suffix="gen-clf-v1"  # Optional suffix for your model name
            )
            job_id = job_response.get('id') if isinstance(
                job_response, dict) else None
            if job_id:
                print(
                    f"Fine-tuning job submitted. Job ID: {job_id}. Polling for completion...")
                fine_tuned_model_id = poll_fine_tuning_job(client, job_id)
                if fine_tuned_model_id:
                    print(
                        f"Successfully obtained fine-tuned model ID: {fine_tuned_model_id}")
                else:
                    print("Could not obtain fine-tuned model ID after polling.")
            else:
                print(
                    f"Failed to submit fine-tuning job or get Job ID. Response: {job_response}")
        except Exception as e:
            print(
                f"An error occurred during fine-tuning submission or polling: {e}")
    else:
        print("Skipping fine-tuning: Training file ID not available.")
    # # Example for testing evaluation part with a known fine-tuned model:
    # # fine_tuned_model_id = "ft:gpt-4o-mini-xxxxxxxxxxxxxxxxxxxxxx"

    # --- 5. List Fine-Tuning Jobs (Optional Check) ---
    print("\n--- Listing Fine-Tuning Jobs ---")
    jobs = list_fine_tuning_jobs(client, limit=5)
    if jobs:
        print("Recent jobs:")
        for job_info in jobs:
            job_status_str = f"  ID: {job_info['id']}, Model: {job_info['model']}, Status: {job_info['status']}"
            if job_info.get('fine_tuned_model'):
                job_status_str += f", Fine-tuned ID: {job_info['fine_tuned_model']}"
            print(job_status_str)

    # --- 6. Evaluate Models ---
    sample_test_items = test_texts[:20]
    sample_test_labels = test_labels[:20]

    # 6a. Evaluate Base Model
    print(f"\n--- Evaluating Base Model ({BASE_MODEL_FOR_FINETUNING}) ---")
    base_model_responses = classify_items_with_model(
        client,
        items_to_classify=sample_test_items,
        model_id=BASE_MODEL_FOR_FINETUNING,
        categories_list=categories,
        prompt_instruction=PROMPT_INSTRUCTION,
        category_list_tag=CATEGORY_LIST_TAG,
        item_wrapper_tag=ITEM_WRAPPER_TAG,
        output_wrapper_tag=OUTPUT_WRAPPER_TAG
    )
    base_model_accuracy = calculate_accuracy(
        base_model_responses, sample_test_labels)
    print(
        f"Base Model ({BASE_MODEL_FOR_FINETUNING}) Accuracy (on {len(sample_test_items)} samples): {base_model_accuracy}%")

    # 6b. Evaluate Fine-Tuned Model
    if fine_tuned_model_id:
        print(f"\n--- Evaluating Fine-Tuned Model ({fine_tuned_model_id}) ---")
        try:
            ft_model_responses = classify_items_with_model(
                client,
                items_to_classify=sample_test_items,
                model_id=fine_tuned_model_id,
                categories_list=categories,
                prompt_instruction=PROMPT_INSTRUCTION,
                category_list_tag=CATEGORY_LIST_TAG,
                item_wrapper_tag=ITEM_WRAPPER_TAG,
                output_wrapper_tag=OUTPUT_WRAPPER_TAG
            )
            ft_model_accuracy = calculate_accuracy(
                ft_model_responses, sample_test_labels)
            print(
                f"Fine-Tuned Model ({fine_tuned_model_id}) Accuracy (on {len(sample_test_items)} samples): {ft_model_accuracy}%")
        except Exception as e:
            print(
                f"Could not evaluate fine-tuned model '{fine_tuned_model_id}'. Error: {e}")
    else:
        print("\nSkipping fine-tuned model evaluation: Fine-tuned model ID not available or job did not complete successfully.")
        print("If you have a fine-tuned model ID from a previous run, you can set 'fine_tuned_model_id' manually for evaluation.")


if __name__ == "__main__":
    main()
