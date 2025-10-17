
import numpy as np
import os
import sys
from openai import AzureOpenAI
import json
import re
import yaml
from pathlib import Path

import requests
import time
import ast
from tqdm import tqdm

from loguru import logger as eval_logger

config = None

NUM_SECONDS_TO_SLEEP = 5

API_TYPE = 'azure'
from concurrent.futures import ThreadPoolExecutor, as_completed

POST_PROCESSING = 'debate'

if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = None
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
else:
    client = AzureOpenAI(
        azure_endpoint = "AZURE_ENDPOINT", 
        api_key="YOUR-API-KEY",  
        api_version="2024-05-01-preview"
    )


# Pass in video path here
# Can only work correctly with video llm
def cvrr_doc_to_visual(doc):
    # Unzip all the zip files to HF HOME cache dir
    HF_HOME = os.environ["HF_HOME"]
    cache_dir = config["dataset_kwargs"]["cache_dir"]
    cache_dir = os.path.join(HF_HOME, cache_dir)
    cache_dir = os.path.join(cache_dir, "CVRR-ES")

    if doc["DimensionName"] == "Continuity and Object Instance Count":
        cache_dir = os.path.join(cache_dir, "continuity_and_object_instance_count")
    elif doc["DimensionName"] == "Fine-grained action understanding":
        cache_dir = os.path.join(cache_dir, "fine_grained_action_understanding")
    elif doc["DimensionName"] == "Interpretation of social context":
        cache_dir = os.path.join(cache_dir, "interpretation_of_social_context")
    elif doc["DimensionName"] == "Interpretation of visual context":
        cache_dir = os.path.join(cache_dir, "interpretation_of_visual_context")
    elif doc["DimensionName"] == "Multiple actions in a single video":
        cache_dir = os.path.join(cache_dir, "multiple_actions_in_a_single_video")
    elif doc["DimensionName"] == "Non-existent actions with existent scene depictions":
        cache_dir = os.path.join(cache_dir, "non_existent_actions_with_existent_scene_depictions")
    elif doc["DimensionName"] == "Non-existent actions with non-existent scene depictions":
        cache_dir = os.path.join(cache_dir, "non_existent_actions_with_non_existent_scene_depictions")
    elif doc["DimensionName"] == "Partial actions":
        cache_dir = os.path.join(cache_dir, "partial_actions")
    elif doc["DimensionName"] == "Time order understanding":
        cache_dir = os.path.join(cache_dir, "time_order_understanding")
    elif doc["DimensionName"] == "Understanding of emotional context":
        cache_dir = os.path.join(cache_dir, "understanding_emotional_context")
    elif doc["DimensionName"] == "Unusual and Physically Anomalous activities":
        cache_dir = os.path.join(cache_dir, "unusual_and_physically_anomalous_activities")

    video_path = doc["VideoID"]
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")

    return [video_path]

def extract_rating(rating_string):
    """
    Extracts the rating number from a string using regex to handle different formats.
    
    Args:
    rating_string (str): The input string containing the rating.
    
    Returns:
    int: The extracted rating number or None if not found.
    """
    # Regex pattern to find rating in different formats
    patterns = [
        r"rating['\"]?\s*:\s*['\"]?(\d+)",   # Matches JSON-like formats with 'rating' key
        r"rating['\"]?\s*:\s*['\"]?\[\[(\d+)\]\]",  # Matches JSON-like formats with rating enclosed in double brackets
        r"rating['\"]?\s*:\s*['\"]?\[(\d+)\]",  # Matches JSON-like formats with rating enclosed in single brackets
        r"^(\d+)\s*\(",                      # Matches formats like "4 (Excellent)"
        r"^(\d+)$",                          # Matches plain number formats like "4"
        r"(\d+)",                             # Any standalone number
    ]

    for pattern in patterns:
        match = re.search(pattern, rating_string)
        if match:
            return int(match.group(1))
    
    return None

def get_gpt_eval(question, answer, pred, max_tokens: int, retries: int = 10):
    global headers

    base_messages = [
        {
            "role": "system",
            "content": "You are an intelligent chatbot designed for evaluating the correctness of AI assistant predictions for question-answer pairs. "
            "Your task is to compare the predicted answer with the ground-truth answer and determine if the predicted answer is correct or not. Here's how you can accomplish the task:"
            "------"
            "##INSTRUCTIONS: "
            "- Focus on the correctness and accuracy of the predicted answer with the ground-truth.\n"
            "- Consider predictions with less specific details as correct evaluation, unless such details are explicitly asked in the question.\n",
        },
        {
            "role": "user",
            "content": "Please evaluate the following video-based question-answer pair:\n\n"
            f"Question: {question}\n"
            f"Ground truth correct Answer: {answer}\n"
            f"Predicted Answer: {pred}\n\n"
            "Provide your evaluation as a correct/incorrect prediction along with the score where the score is an integer value between 1 (fully wrong) and 5 (fully correct). Here is the detailed scoring rubric for evaluating:\nPoor (1): The predicted answer significantly deviates from the user's instruction and fails to address the query effectively. It shows a lack of relevance, accuracy, and comprehensiveness. Creativity and granularity are absent or poorly executed.\nFair (2): The predicted answer addresses the user's instruction partially, with evident shortcomings in relevance, accuracy, or comprehensiveness. It lacks depth in creativity and granularity, indicating a superficial understanding of the user's inquiry.\nAverage (3): The predicted answer adequately addresses the user's instruction, showing a fair level of relevance, accuracy, and comprehensiveness. It reflects a basic level of creativity and granularity but may lack sophistication or depth in fully capturing the user's inquiry.\nGood (4): The predicted answer is well-aligned with the user's instruction, demonstrating a high degree of relevance, accuracy, and comprehensiveness. It shows creativity and a nuanced understanding of the topic, with detailed granularity that enhances the quality.\nExcellent (5): The predicted answer perfectly adheres to the user's instruction, excelling in relevance, accuracy, comprehensiveness, creativity, and granularity. It provides an insightful, detailed, and thorough answer, indicating a deep and nuanced understanding of the user's inquiry."
            "Please generate the response in the form of a Python dictionary string with keys 'pred', 'score' and 'reason', where value of 'pred' is a string of 'correct' or 'incorrect', value of 'score' is in INTEGER, not STRING and value of 'reason' should provide the reason behind the decision."
            "Only provide the Python dictionary string."
            'For example, your response should look like this: {"pred": "correct", "rating": 4, "reason": reason}.',
        },
    ]

    for attempt in range(retries):
        try:
            # Agent 1 initial evaluation
            response1 = client.chat.completions.create(
                model="gpt35",
                messages=base_messages,
                max_tokens=max_tokens,
                temperature=0,
                logprobs=False
            )

            # Agent 2 initial evaluation
            response2 = client.chat.completions.create(
                model="gpt4o",
                messages=base_messages,
                max_tokens=max_tokens,
                temperature=0,
                logprobs=False
            )

            try:
                response_data1 = response1.to_dict()
                response_data2 = response2.to_dict()
            except requests.exceptions.JSONDecodeError:
                eval_logger.error(f"JSON decode error on attempt {attempt + 1}.")
                continue

            content1 = response_data1["choices"][0]["message"]["content"].strip()
            content2 = response_data2["choices"][0]["message"]["content"].strip()

            rating1, rating2 = extract_rating(content1), extract_rating(content2)
            print(f"rating1:{rating1}")
            print(f"rating2:{rating2}")
            print("=" * 64)
            if rating1 == rating2:
                return content2, response_data1["model"]
            
            if content1 != "" and content2 != "":
                # Debate between agents
                debate_messages = base_messages + [
                    {"role": "assistant", "content": f"Agent 1 evaluation: {content1}"},
                    {"role": "assistant", "content": f"Agent 2 evaluation: {content2}"},
                    {"role": "user", "content": "Agent 1, please respond to Agent 2's evaluation. Do you agree or disagree? Provide reasoning for your stance."}
                ]

                # Agent 1 response to Agent 2
                response3 = client.chat.completions.create(
                    model="gpt35",
                    messages=debate_messages,
                    max_tokens=max_tokens,
                    temperature=0,
                    logprobs=False
                )

                content3 = response3.to_dict()["choices"][0]["message"]["content"].strip()
                debate_messages.append({"role": "assistant", "content": f"Agent 1 response: {content3}"})
                debate_messages.append({"role": "user", "content": "Agent 2, considering Agent 1's response, do you want to modify your evaluation? If so, provide your updated evaluation."})

                # Agent 2 final response
                response4 = client.chat.completions.create(
                    model="gpt4o",
                    messages=debate_messages,
                    max_tokens=max_tokens,
                    temperature=0,
                    logprobs=False
                )

                content4 = response4.to_dict()["choices"][0]["message"]["content"].strip()
                debate_messages.append({"role": "assistant", "content": f"Agent 2 final response: {content4}"})
                debate_messages.append({"role": "user", "content": "Based on the debate between Agent 1 and Agent 2, please provide a final consensus evaluation. Consider both perspectives and provide a well-reasoned final decision. Please generate the response in the form of a Python dictionary string with keys 'pred', 'score' and 'reason', where value of 'pred' is a string of 'correct' or 'incorrect', value of 'score' is in INTEGER, not STRING and value of 'reason' should provide the reason behind the decision. Only provide the Python dictionary string. For example, your response should look like this: {'pred': 'correct', 'rating': 4, 'reason': 'reason'}."})
                
                # Final consensus
                final_response = client.chat.completions.create(
                    model="gpt4o",
                    messages=debate_messages,
                    max_tokens=max_tokens,
                    temperature=0,
                    logprobs=False
                )

                final_content = final_response.to_dict()["choices"][0]["message"]["content"].strip()
                return final_content, response_data1["model"]

        except requests.exceptions.HTTPError as e:
            eval_logger.error(f"HTTP error on attempt {attempt + 1}: {e}")
        except requests.exceptions.RequestException as e:
            eval_logger.error(f"Request exception on attempt {attempt + 1}: {e}")
        except Exception as e:
            eval_logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
            if e['error']['code'] == '429':
                return content2, response_data1["model"]


        if attempt < retries - 1:
            time.sleep(NUM_SECONDS_TO_SLEEP)
        else:
            eval_logger.error(f"All {retries} attempts failed. Last error message: {e}")
            return "", ""

    return "", ""

import ast
import json
import re

def parse_score(review):
    """
    Parses a review string to extract correctness, score, and reason.

    Args:
        review (str): The review string to parse.

    Returns:
        tuple: A tuple containing correctness (str), score (int), and reason (str).
    """
    # Step 1: Remove code fences if present
    review = review.strip()
    code_fence_pattern = r'^```(?:\w+)?\n(.*?)\n```$'
    match = re.match(code_fence_pattern, review, re.DOTALL)
    if match:
        review = match.group(1).strip()

    # Step 2: Attempt to parse using ast.literal_eval (handles single and double quotes)
    try:
        review_dict = ast.literal_eval(review)
    except (ValueError, SyntaxError):
        # Step 3: If ast.literal_eval fails, attempt to fix the string for JSON parsing
        try:
            # Replace single quotes around keys and string values with double quotes
            # This regex ensures that only quotes around keys and values are replaced
            review_fixed = re.sub(r"(?<!\\)'(\w+)'(?s):", r'"\1":', review)      # Keys
            review_fixed = re.sub(r":\s*'([^']*)'", r': "\1"', review_fixed)   # String values
            review_dict = json.loads(review_fixed)
        except json.JSONDecodeError:
            # Step 4: If JSON parsing also fails, return default values
            return "incorrect", 0, ""

    # Step 5: Extract required fields with default values
    correctness = review_dict.get("pred", "incorrect")
    score = review_dict.get("score", 0)
    reason = review_dict.get("reason", "")

    # Ensure score is an integer
    try:
        score = int(score)
    except (ValueError, TypeError):
        score = 0

    return correctness, score, reason


# Process result for evaluation in temporal task
def cvrr_process_results(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary
    """
    try:
        question = doc["Q"]
        answer = doc["A"]
        pred = result[0]

        # Assume get_gpt_eval returns a review and the model name, and parse_score parses this review
        review, model_name = get_gpt_eval(question, answer, pred, 512)
        correctness, score, reason = parse_score(review)
    except Exception as e:
        eval_logger.error(f"Error for Question ID: {doc.get('question_id', 'Unknown')}: {e}")
        review = "Failed to Get a Proper Review."
        model_name = "Failed Request"
        score = 0
        correctness = "incorrect"
        reason = ""

    return {
        "gpt_eval_score": {"VideoID": doc["VideoID"], "Q": doc["Q"], "A": doc["A"], "pred": pred, "DimensionName": doc["DimensionName"], "correctness": correctness, "score": score, "reason": reason},
        "gpt_eval_accuracy": {"VideoID": doc["VideoID"], "Q": doc["Q"], "A": doc["A"], "pred": pred, "DimensionName": doc["DimensionName"], "correctness": correctness, "score": score, "reason": reason},
    }


# Factory into different aggregate
def cvrr_aggregate_score(results, args):
    total_score = 0

    # Iterate over the results to sum scores
    for result_dict in results:
        total_score += result_dict["score"]

    # Calculate average score
    average_score = total_score / len(results) if results else 0
    eval_logger.info(f"Average Score: {average_score}")
    return average_score


def cvrr_aggregate_accuracy(results, args):
    yes_count = 0
    no_count = 0

    # Iterate over the results to count correctness
    for result_dict in results:
        if result_dict["correctness"] == "correct":
            yes_count += 1
        else:
            no_count += 1

    # Calculate accuracy and average score
    accuracy = yes_count / (yes_count + no_count) if (yes_count + no_count) > 0 else 0
    eval_logger.info(f"Accuracy: {accuracy}")
    return accuracy * 100


# # Define the list of candidate models and judge models
# candidate_model = ["llama_vid", "gpt4v", "video_chatgpt", "mplug_owl_Video", "video_llava"]
# judge_model = ['video_llava', 'llama_vid', 'gpt4v', 'internvl2']  # Assuming 'mplug_owl_Video' is not a judge

# import json
# # gpt3.5 score
# input_file = "../batch_with_score.json"
# with open(input_file, 'r') as f:
#     data = json.load(f)

# results = {}
# for dataset, samples in data.items():
#     results[dataset] = {'gpt3.5': {candidate: [] for candidate in candidate_model}}
#     for candidate in candidate_model:
#         input_file = f"{dataset}/{candidate}/{dataset}.json"
#         with open(input_file, 'r') as f:
#             candidate_data = json.load(f)
#         print(input_file)
#         for sample_data in candidate_data['logs']:
#             doc = sample_data['doc']
#             result = sample_data['filtered_resps']
#             update = cvrr_process_results(doc, result)
#             sample_data.update(update)
#             with open(input_file + '_debate.json', 'w') as f:
#                 json.dump(candidate_data, f, indent=4)


# some sample failed, we need to correct them
# results = {}
# for dataset, samples in data.items():
#     results[dataset] = {'gpt3.5': {candidate: [] for candidate in candidate_model}}
#     for candidate in candidate_model:
#         input_file = f"{dataset}/{candidate}/{dataset}.json"
#         with open(input_file, 'r') as f:
#             candidate_data = json.load(f)
#         with open(input_file + 'newnew.json', 'r') as f:
#             new_candidate_data = json.load(f)
#         print(input_file)
#         for sample_data in new_candidate_data['logs']:
#             if len(sample_data['gpt_eval_score']['reason']) and sample_data['gpt_eval_score']['score'] != 0:
#                 continue
#             doc = sample_data['doc']
#             result = sample_data['filtered_resps']
#             update = cvrr_process_results(doc, result)
#             sample_data.update(update)
#         with open(input_file + 'newnew.json', 'w') as f:
#             json.dump(new_candidate_data, f, indent=4)


# collect the results
# results = {}
# for dataset, samples in data.items():
#     results[dataset] = {'gpt3.5': {candidate: [] for candidate in candidate_model}}
#     for candidate in candidate_model:
#         input_file = f"{dataset}/{candidate}/{dataset}.json"
#         with open(input_file + 'newnew.json', 'r') as f:
#             new_candidate_data = json.load(f)
#         print(input_file)
#         for sample_data in new_candidate_data['logs']:
#             doc_id = str(sample_data['doc_id'])
#             score = sample_data['gpt_eval_score']['score']
#             data[dataset][doc_id][candidate][-1] = score
# with open('../batch_with_score.json', 'w') as f:
#     json.dump(data, f, indent=4)



# Define the list of candidate models and judge models
candidate_model = ["llama_vid", "gpt4v", "video_chatgpt", "mplug_owl_Video", "video_llava"]
judge_model = ['video_llava', 'llama_vid', 'gpt4v', 'internvl2']  # Assuming 'mplug_owl_Video' is not a judge

# gpt3.5 score
input_file = "../batch_with_score.json"
with open(input_file, 'r') as f:
    data = json.load(f)

def process_candidate(dataset, candidate):
    input_file_path = f"{dataset}/{candidate}/{dataset}.json"
    if POST_PROCESSING:
        input_file_path = f"{input_file_path}_debate.json"
    debate_file_path = f"{input_file_path}_debate{POST_PROCESSING}.json"
    input_file_path = debate_file_path
    try:
        with open(input_file_path, 'r') as f:
            candidate_data = json.load(f)
    except FileNotFoundError:
        eval_logger.error(f"File not found: {input_file_path}")
        return

    eval_logger.info(f"Processing file: {input_file_path}")

    # Use tqdm for progress bar within each candidate processing
    for sample_data in tqdm(candidate_data.get('logs', []), desc=f"{dataset}/{candidate}"):
        doc = sample_data.get('doc')
        result = sample_data.get('filtered_resps')
        if not doc or not result:
            eval_logger.warning(f"Missing 'doc' or 'filtered_resps' in sample_data: {sample_data}")
            continue
        if POST_PROCESSING and len(sample_data['gpt_eval_score']['reason']) and sample_data['gpt_eval_score']['score'] != 0:
            continue
        update = cvrr_process_results(doc, result)
        sample_data.update(update)

    # Write the updated data to the debate file
    with open(debate_file_path, 'w') as f:
        json.dump(candidate_data, f, indent=4)

    eval_logger.info(f"Completed processing for {input_file_path}. Results saved to {debate_file_path}")


# Initialize ThreadPoolExecutor with a suitable number of workers
# You can adjust max_workers based on your system's capabilities and API rate limits
max_workers = 10
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Prepare a list of all (dataset, candidate) pairs
    tasks = []
    for dataset in data.keys():
        for candidate in candidate_model:
            tasks.append((dataset, candidate))

    # Submit all tasks to the executor
    future_to_task = {executor.submit(process_candidate, dataset, candidate): (dataset, candidate) for dataset, candidate in tasks}

    # Optionally, use as_completed to handle results as they finish
    for future in as_completed(future_to_task):
        dataset, candidate = future_to_task[future]
        try:
            future.result()
        except Exception as exc:
            eval_logger.error(f"Generated an exception for {dataset}/{candidate}: {exc}")
        else:
            eval_logger.info(f"Successfully processed {dataset}/{candidate}")