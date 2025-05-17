import json
import os
from typing import List, Dict, Any, Tuple, Optional

from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv

def load_classification_data(
        train_file_path: str,
        test_file_path: str,
        sep: str = "\t",
        text_column: str = "text",
        label_column: str = "label",
)-> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str], List[str], List[str]]:
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
        ticket_text: str, 
        categories_list: List[str]
    ) -> str:
    """
    Helper function to create a standardized prompt for classification.
    """
    categories_str = '\n'.join(categories_list)
    return f"""classify a customer support ticket into one of the following categories:
            <categories>
            {categories_str}
            </categories>

            Here is the customer support ticket:
            <ticket>{ticket_text}</ticket>

            Respond using this format:
            <category>The category label you chose goes here</category>
            """

def format_data_for_openai_chat_completions(
    dataframe: pd.DataFrame,
    text_column: str,
    label_column: str,
    categories_list: List[str]
) -> List[Dict[str, List[Dict[str, str]]]]:
    """
    Converts a DataFrame into the JSONL format expected by OpenAI for fine-tuning
    chat completion models.

    Args:
        dataframe (pd.DataFrame): DataFrame containing the text and label columns.
        text_column (str): Name of the column with input text.
        label_column (str): Name of the column with labels.
        categories_list (List[str]): List of possible categories for the prompt.

    Returns:
        List[Dict[str, List[Dict[str, str]]]]: A list of message objects.
    """
    json_objs = []
    for _, example in dataframe.iterrows():
        user_msg_content = _create_classification_prompt(example[text_column], categories_list)
        assistant_msg_content = f"<category>{example[label_column]}</category>"
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
    model_name: str, # e.g., 'gpt-4o-mini-2024-07-18' or 'gpt-3.5-turbo'
    batch_size: Optional[Any] = "auto", # Can be int or "auto"
    learning_rate_multiplier: Optional[Any] = "auto", # Can be float or "auto"
    n_epochs: Optional[Any] = "auto", # Can be int or "auto", default is often 3 for gpt-3.5-turbo
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
    print(f"Starting fine-tuning job for model '{model_name}' with training file ID '{training_file_id}'.")
    print("Be mindful of the costs associated with fine-tuning.")

    hyperparameters = {
        "batch_size": batch_size,
        "learning_rate_multiplier": learning_rate_multiplier,
        "n_epochs": n_epochs,
    }
    # Filter out None values to use OpenAI defaults if not specified
    filtered_hyperparameters = {k: v for k, v in hyperparameters.items() if v is not None}


    job_payload = {
        "training_file": training_file_id,
        "model": model_name,
    }
    # The new fine_tuning.jobs.create API uses a 'hyperparameters' dictionary
    # For older models or different API versions, this structure might vary.
    # The provided script uses a nested structure for `method` and `supervised`.
    # As of early 2024, for models like gpt-3.5-turbo, the structure is flatter.
    # For gpt-4o-mini fine-tuning:
    if "gpt-4o-mini" in model_name or "gpt-4" in model_name or "gpt-3.5" in model_name: # Newer models
         job_payload["hyperparameters"] = filtered_hyperparameters
    else: # Fallback or for older models that might use a different structure.
          # The script provided used a 'method' field, which is for the older API.
          # The current FineTuningJob.create uses 'hyperparameters' directly.
          # If your specific model version needs the "method" and "supervised" structure,
          # you might need to adjust this part.
          # For now, we'll assume the newer structure.
          # If the user provided the exact structure from the notebook:
          # method = {
          #   "type": "supervised",
          #   "supervised": {
          #     "hyperparameters": filtered_hyperparameters
          #   }
          # }
          # job_payload["method"] = method # This is likely for an older API version
          # Let's stick to the current documented way for `client.fine_tuning.jobs.create`
          job_payload["hyperparameters"] = filtered_hyperparameters


    if suffix:
        job_payload["suffix"] = suffix

    try:
        job = client.fine_tuning.jobs.create(**job_payload)
        print(f"Fine-tuning job created successfully. Job ID: {job.id}, Status: {job.status}")
        return job.to_dict() # Return as a dictionary for easier handling
    except Exception as e:
        print(f"Error creating fine-tuning job: {e}")
        raise

def list_fine_tuning_jobs(client: OpenAI, limit: int = 20) -> List[Dict[str, Any]]:
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

def retrieve_fine_tuning_job(client: OpenAI, job_id: str) -> Dict[str, Any]:
    """
    Retrieves a specific fine-tuning job by its ID.

    Args:
        client (OpenAI): The initialized OpenAI client.
        job_id (str): The ID of the fine-tuning job to retrieve.

    Returns:
        Dict[str, Any]: The fine-tuning job object.
    """
    print(f"Retrieving fine-tuning job with ID: {job_id}")
    try:
        job = client.fine_tuning.jobs.retrieve(job_id)
        print(f"Job Status: {job.status}")
        if job.fine_tuned_model:
            print(f"Fine-tuned Model ID: {job.fine_tuned_model}")
        return job.to_dict()
    except Exception as e:
        print(f"Error retrieving fine-tuning job {job_id}: {e}")
        raise

def classify_tickets_with_model(
    client: OpenAI,
    tickets_to_classify: List[str],
    model_id: str,
    categories_list: List[str],
    temperature: float = 0.0,
    stop_sequence: Optional[List[str]] = None, # Changed to List[str]
    max_tokens: int = 50 # Reduced for classification, as only category is expected
) -> List[str]:
    """
    Classifies a list of support tickets using a specified OpenAI model.

    Args:
        client (OpenAI): The initialized OpenAI client.
        tickets_to_classify (List[str]): A list of ticket texts.
        model_id (str): The ID of the OpenAI model to use (e.g., "gpt-4o-mini" or a fine-tuned model ID).
        categories_list (List[str]): List of possible categories for the prompt.
        temperature (float): Sampling temperature. Lower is more deterministic.
        stop_sequence (Optional[List[str]]): Sequence(s) where the API will stop generating further tokens.
                                         The example used "</category>".
        max_tokens (int): Maximum number of tokens to generate for the completion.

    Returns:
        List[str]: A list of predicted category labels.
    """
    if stop_sequence is None:
        stop_sequence = ["</category>"] # Default stop sequence

    responses = []
    print(f"Classifying {len(tickets_to_classify)} tickets using model: {model_id}")
    for i, ticket in enumerate(tickets_to_classify):
        user_prompt_content = _create_classification_prompt(ticket, categories_list)
        try:
            completion = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": user_prompt_content}],
                temperature=temperature,
                stop=stop_sequence,
                max_tokens=max_tokens,
            )
            response_content = completion.choices[0].message.content
            # Extract content within <category>...</category>
            if "<category>" in response_content:
                predicted_category = response_content.split("<category>")[-1].strip()
            else: # Fallback if the model doesn't perfectly follow the format
                predicted_category = response_content.strip()
            
            responses.append(predicted_category)
            print(f"Ticket {i+1}/{len(tickets_to_classify)} classified as: {predicted_category}")
        except Exception as e:
            print(f"Error classifying ticket {i+1}: {ticket[:50]}... - Error: {e}")
            responses.append("CLASSIFICATION_ERROR") # Placeholder for errors
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
        raise ValueError("Predicted and actual labels lists must have the same length.")
    if not actual_labels: # Avoid division by zero for empty lists
        return 0.0
    
    num_correct = sum(p == a for p, a in zip(predicted_labels, actual_labels))
    accuracy = round(100 * num_correct / len(actual_labels), 2)
    print(f"Accuracy calculated: {accuracy}% ({num_correct}/{len(actual_labels)})")
    return accuracy

if __name__ == "__main__":
    # --- Configuration ---
    TRAIN_FILE = '../data/train.tsv'  
    TEST_FILE = '../data/test.tsv'    
    OUTPUT_JSONL_FILE = 'ticket_classification_training_data.jsonl'
    
    BASE_MODEL_FOR_FINETUNING = 'gpt-4o-mini-2024-07-18' 

    # --- Initialize ---
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # --- 1. Load and Prepare Data ---
    (training_df, test_df,
     training_texts, training_labels,
     test_texts, test_labels,
     categories) = load_classification_data(TRAIN_FILE, TEST_FILE)

    print(f"\nTraining examples head:\n{training_df.head()}")

    # --- 2. Format Data for OpenAI ---
    training_json_data = format_data_for_openai_chat_completions(
        training_df, 'text', 'label', categories
    )
    save_to_jsonl(training_json_data, OUTPUT_JSONL_FILE)

    # --- 3. Upload Training File ---
    print("\n--- Uploading Training File ---")
    training_file_id = upload_file_to_openai(client, OUTPUT_JSONL_FILE)
    print(f"Training File ID for fine-tuning: {training_file_id}")
    # Example: training_file_id = "file-xxxxxxxxxxxxxxxxxxxxx" # Replace with your actual file ID

    # --- 4. Start Fine-Tuning Job (THIS WILL INCUR COSTS) ---
    print("\n--- Starting Fine-Tuning Job ---")
    # Make sure training_file_id is set from the previous step's output
    if 'training_file_id' in locals():
        fine_tuning_job = start_fine_tuning_job(
            client,
            training_file_id=training_file_id, 
            model_name=BASE_MODEL_FOR_FINETUNING,
            n_epochs=3, 
            suffix="finetuning_clf"
        )
        print(f"Fine-tuning job details: {fine_tuning_job}")
        # You will need to monitor the job status via list_fine_tuning_jobs or retrieve_fine_tuning_job
        # The fine_tuned_model ID will be available once the job is 'succeeded'.
    else:
        print("Skipping fine-tuning job creation as training_file_id is not set.")
        print("To run this step, uncomment file upload and fine-tuning job creation sections.")

    # --- 5. List and Retrieve Fine-Tuning Jobs (Example) ---
    print("\n--- Listing Fine-Tuning Jobs ---")
    jobs = list_fine_tuning_jobs(client, limit=5)
    if jobs:
        print("Recent jobs:")
        for job_info in jobs:
            print(f"  ID: {job_info['id']}, Model: {job_info['model']}, Status: {job_info['status']}, Fine-tuned model: {job_info.get('fine_tuned_model')}")
        
        # Example: Retrieve details for the first job in the list if it exists
        first_job_id = jobs[0]['id']
        job_details = retrieve_fine_tuning_job(client, first_job_id)
        print(f"Details for job {first_job_id}: {job_details}")


    # --- 6. Evaluate Models ---
    sample_test_tickets = test_texts[:10]
    sample_test_labels = test_labels[:10]

    # 6a. Evaluate Base Model
    print("\n--- Evaluating Base Model ---")
    base_model_id = BASE_MODEL_FOR_FINETUNING # Or another base model like 'gpt-4o-mini'
    base_model_responses = classify_tickets_with_model(
        client,
        tickets_to_classify=sample_test_tickets,
        model_id=base_model_id,
        categories_list=categories
    )
    base_model_accuracy = calculate_accuracy(base_model_responses, sample_test_labels)
    print(f"Base Model ({base_model_id}) Test Set Accuracy (on {len(sample_test_tickets)} samples): {base_model_accuracy}%")

    # 6b. Evaluate Fine-Tuned Model
    print("\n--- Evaluating Fine-Tuned Model ---")
    # IMPORTANT: Replace with your actual fine-tuned model ID once training is complete and successful.
    # It will look something like: "ft:gpt-4o-mini-2024-07-18:your-org:suffix:xxxxxxxx"
    # You can get this from the `retrieve_fine_tuning_job` output or the OpenAI dashboard.
    fine_tuned_model_id = "YOUR_FINE_TUNED_MODEL_ID_HERE" # <--- REPLACE THIS

    if fine_tuned_model_id != "YOUR_FINE_TUNED_MODEL_ID_HERE":
        try:
            ft_model_responses = classify_tickets_with_model(
                client,
                tickets_to_classify=sample_test_tickets,
                model_id=fine_tuned_model_id,
                categories_list=categories
            )
            ft_model_accuracy = calculate_accuracy(ft_model_responses, sample_test_labels)
            print(f"Fine-Tuned Model ({fine_tuned_model_id}) Test Set Accuracy (on {len(sample_test_tickets)} samples): {ft_model_accuracy}%")
        except Exception as e:
            print(f"Could not evaluate fine-tuned model. Is the ID '{fine_tuned_model_id}' correct and the model ready? Error: {e}")
    else:
        print("Skipping fine-tuned model evaluation. Please replace 'YOUR_FINE_TUNED_MODEL_ID_HERE' with your actual model ID.")

    print("\n--- Script Finished ---")