import os
import time
import random
import itertools
import json
import collections
import sys
import inspect
from tqdm import tqdm
import json
import torch
import copy

import numpy as np
from datasets import Image, Sequence

import lmms_eval.api
import lmms_eval.tasks
import lmms_eval.models
import lmms_eval.api.metrics
import lmms_eval.api.registry

from lmms_eval.utils import (
    positional_deprecated,
    run_task_tests,
    make_table,
    create_iterator,
    get_git_commit_hash,
    simple_parse_args_string,
)

from loguru import logger as eval_logger

def get_prompt():
    # some of prompts are from https://github.com/Dongping-Chen/MLLM-Judge
    prompt = {
        "start": """Please serve as an unbiased judge in assessing the quality of the responses from AI assistants regarding the user's instruction and a video. """,
        "setting": {"COT figure": """Please examine the provided video attentively. Begin by conducting a comprehensive analysis of the figure provided. Detail your observations and insights in the 'Figure Analysis' section. Next, utilize the insights from your initial analysis to critically evaluate the responses. Summarize this evaluation in the 'Analysis' section. Finally, based on your figure analysis and response evaluation, form a well-reasoned judgement. Document this in the 'Judgement' section. Ensure that your final output with keys: 'Figure Analysis' for the initial figure assessment, 'Analysis' for the evaluation of responses against your ground truth, and 'Judgement' for your final judgement only. Ensure that the content under each key does not contain any nested JSON structures.""",
            "COT instruction": """Please examine the provided video attentively. Begin by providing a rating number. Then provide detailed response to the user instructions, treating this response as the baseline or 'ground truth'. This response will form the 'Response' section. Next, use this established ground truth to systematically analyze and evaluate the responses to the same instruction. This evaluation will form the 'Analysis' section. After the analysis, move forward to the judgement phase, where you will give final judgement based on the analysis of the responses compared to the ground truth. Give your judgement in the 'Judgement' section. Ensure that your final output is structured with the keys 'Response' for the answer to the instruction, 'Analysis' for the evaluation of responses, and 'Judgement' for your final judgement only. Ensure that the content under each key does not contain any nested JSON structures.""",
            "COT figure instruction": """Please examine the provided video attentively. Begin with an in-depth analysis of the figure. Detail your observations and insights in the 'Figure Analysis' section. Then, provide a detailed response to the user instructions, treating this as 'Response' and the ground truth. Next, compare and analyze the responses to the same instruction against your ground truth in the 'Analysis' section. Finally, give your final judgement in 'Judgement'. Structure your output in JSON format, with the following keys: 'Figure Analysis' for the initial figure assessment, 'Response' for your response to the instructions, 'Analysis' for the evaluation of responses against your ground truth, and 'Judgement' for your final judgement only. Ensure that the content under each key does not contain any nested JSON structures.""",
            "No COT": """Please examine the provided video attentively. Begin by conducting a detailed analysis of the responses provided. Capture your comprehensive observations and insights in the 'Analysis' section. Following your analysis, move on to the judgement phase, where you will make informed decisions or conclusions based on the analysis conducted. Give your final judgements in the 'Judgement' section. Ensure that your final output in a JSON format with keys 'Analysis' for the initial response analysis, and 'Judgement' for your final judgement only. Ensure that the content under each key does not contain any nested JSON structures.""",
            "No Figure": """As a blind judge, you will not have access to the figure mentioned in the user instructions. Your task is to impartially assess the responses based solely on the information presented within them, without visual context of the figure. Begin by performing a detailed analysis of the responses, capturing your observations in the 'Analysis' section. Then, move on to the judgement phase, drawing conclusions or making decisions based on your analysis. Format your findings in a JSON format with two keys: 'Analysis' for your insights on the responses and 'Judgement' for your final judgement only. Ensure that the content under each key does not contain any nested JSON structures.""",
            "Vision Expert": """As a blind judge, you won't receive the figure from the user instructions, lacking direct visual context. An AI-generated analysis will be provided as optional supplementary information, but bear in mind its potential inaccuracies. Your primary task is to conduct a thorough analysis of the responses independently. Include your observations and interpretations in the 'Analysis' section. Following this, advance to the judgement phase, forming decisions based on your analysis, optionally informed by the AI analysis. Present your findings in a JSON format with keys 'Analysis' for your insights on the responses and 'Judgement' for your final judgement only. Ensure that the content under each key does not contain any nested JSON structures."""},
        "tasks": {"score": """
You will receive a single response from the AI assistant to user's instruction. Use scores to show the quality of the response. Here is the detailed scoring rubric for evaluating the quality of responses from AI assistants:
Poor (1): The response significantly deviates from the user's instruction and fails to address the query effectively. It shows a lack of relevance, accuracy, and comprehensiveness. Creativity and granularity are absent or poorly executed.
Fair (2): The response addresses the user's instruction partially, with evident shortcomings in relevance, accuracy, or comprehensiveness. It lacks depth in creativity and granularity, indicating a superficial understanding of the user's inquiry.
Average (3): The response adequately addresses the user's instruction, showing a fair level of relevance, accuracy, and comprehensiveness. It reflects a basic level of creativity and granularity but may lack sophistication or depth in fully capturing the user's inquiry.
Good (4): The response is well-aligned with the user's instruction, demonstrating a high degree of relevance, accuracy, and comprehensiveness. It shows creativity and a nuanced understanding of the topic, with detailed granularity that enhances the response quality.
Excellent (5): The response perfectly adheres to the user's instruction, excelling in relevance, accuracy, comprehensiveness, creativity, and granularity. It provides an insightful, detailed, and thorough answer, indicating a deep and nuanced understanding of the user's inquiry.

Use "[[1]]", "[[2]]", "[[3]]", "[[4]]", "[[5]]" to indicate your evaluate score in the key 'Judgement'.
""",
"pair": """You will be presented with two responses from different assistants to the same user instruction.
Your task is to assess and compare these responses based on how effectively they adhere to the user's original instruction and how aptly they address the user's inquiry.

Indicate your decision in the key 'Judgement', use "[[A]]" if assistant A prevails, "[[B]]" if assistant B does, and "[[C]]" for a tie.
""",
"batch": """You will be presented with several responses from different assistants to the same user instruction.
Your task is to assess and  compare these responses based on how effectively they adhere to the user's original instruction and how aptly they address the user's inquiry.
After your assessment and comparison, you should RANK the responses from best to worst as the following template. If Assistant A is the best response, Assistant D is the worst response, you should output like [[A]], [[B]], [[C]], [[D]]. Indicate your final rank in the key 'Judgement'."""},
        "notice": """Your assessment should identify whether the assistant effectively adheres to the user's instruction and addresses the user's inquiry.
In your evaluation, weigh factors such as relevance, accuracy, comprehensiveness, creativity, and the granularity of the responses.
Do not allow the length of the responses to influence your evaluation.
Do not favor certain names or position of the assistants. Be as objective as possible."""
    }
    return prompt

def construct_input(prompt_dict, judge_mode, reference, instruction, responses):
    prompt_dict["tasks"]["score_reference"] = prompt_dict["tasks"]["score"]
    prompt = prompt_dict["start"] + "\nEvaluation Method:\n" + prompt_dict["tasks"][judge_mode] + "\nHere is the input:\n"
    if judge_mode == "score":
        prompt += f"""
[The Start of User Instruction]
{instruction}
[The End of User Instruction]
[The Start of Assistant’s Answer]
{responses[0]}
[The End of Assistant’s Answer]""" 
    elif judge_mode == "score_reference":
        prompt += f"""
[The Start of User Instruction]
{instruction}
[The End of User Instruction]
[The Start of Assistant’s Answer]
{responses[0]}
[The End of Assistant’s Answer]
Ratings from other judges may be used as additional information:\n""" 
        judge_name = "A"
        for i in range(len(reference)):
            prompt += f"[The Start of Judge {judge_name}’s rating]\n"
            prompt += reference[i] + "\n"
            prompt += f"[The End of Judge {judge_name}’s rating]\n"
            judge_name = chr(ord(judge_name) + 1)
    elif judge_mode == 'pair':
        prompt += f"""
[The Start of User Instruction]
{instruction}
[The End of User Instruction]
[The Start of Assistant A’s Answer]
{responses[0]}
[The End of Assistant A’s Answer]
[The Start of Assistant B’s Answer]
{responses[1]}
[The End of Assistant B’s Answer]"""
    elif judge_mode == 'batch':
        prompt += f"""
[The Start of User Instruction]
{instruction}
[The End of User Instruction]"""
        assistant_name = "A"
        num_assistant = 0
        for i in range(len(responses)):
            prompt += f"[The Start of Assistant {assistant_name}’s Answer]\n"
            prompt += responses[num_assistant] + "\n"
            prompt += f"[The End of Assistant {assistant_name}’s Answer]\n"
            assistant_name = chr(ord(assistant_name) + 1)
            num_assistant += 1
    return prompt


cvrr_scores_dict = {
    "video_llava": {
        "cvrr_continuity_and_object_instance_count": 0.0135,
        "cvrr_fine_grained_action_understanding": 0.0052,
        "cvrr_interpretation_of_social_context": 0.0215,
        "cvrr_interpretation_of_visual_context": 0.0281,
        "cvrr_multiple_actions_in_a_single_video": 0.0050,
        "cvrr_non_existent_actions_with_existent_scene_depictions": -0.0183,
        "cvrr_non_existent_actions_with_non_existent_scene_depictions": -0.0138,
        "cvrr_partial_actions": 0.0047,
        "cvrr_time_order_understanding": 0.0068,
        "cvrr_understanding_emotional_context": 0.0023,
        "cvrr_unusual_and_physically_anomalous_activities": 0.0037
    },
    "llama_vid": {
        "cvrr_continuity_and_object_instance_count": 0.0370,
        "cvrr_fine_grained_action_understanding": 0.0201,
        "cvrr_interpretation_of_social_context": 0.0104,
        "cvrr_interpretation_of_visual_context": 0.0308,
        "cvrr_multiple_actions_in_a_single_video": 0.0139,
        "cvrr_non_existent_actions_with_existent_scene_depictions": -0.0415,
        "cvrr_non_existent_actions_with_non_existent_scene_depictions": -0.0404,
        "cvrr_partial_actions": 0.0200,
        "cvrr_time_order_understanding": 0.0299,
        "cvrr_understanding_emotional_context": 0.0047,
        "cvrr_unusual_and_physically_anomalous_activities": -0.0025
    },
    "gpt4v": {
        "cvrr_continuity_and_object_instance_count": 0.1302,
        "cvrr_fine_grained_action_understanding": 0.2242,
        "cvrr_interpretation_of_social_context": 0.1867,
        "cvrr_interpretation_of_visual_context": 0.1810,
        "cvrr_multiple_actions_in_a_single_video": 0.1683,
        "cvrr_non_existent_actions_with_existent_scene_depictions": -0.0276,
        "cvrr_non_existent_actions_with_non_existent_scene_depictions": -0.1249,
        "cvrr_partial_actions": 0.0082,
        "cvrr_time_order_understanding": 0.0733,
        "cvrr_understanding_emotional_context": 0.1171,
        "cvrr_unusual_and_physically_anomalous_activities": 0.0606
    },
    "internvl2": {
        "cvrr_continuity_and_object_instance_count": 0.1069,
        "cvrr_fine_grained_action_understanding": 0.1347,
        "cvrr_interpretation_of_social_context": 0.1208,
        "cvrr_interpretation_of_visual_context": 0.1671,
        "cvrr_multiple_actions_in_a_single_video": 0.0952,
        "cvrr_non_existent_actions_with_existent_scene_depictions": 0.0218,
        "cvrr_non_existent_actions_with_non_existent_scene_depictions": 0.0309,
        "cvrr_partial_actions": 0.0321,
        "cvrr_time_order_understanding": 0.0933,
        "cvrr_understanding_emotional_context": 0.0750,
        "cvrr_unusual_and_physically_anomalous_activities": 0.0610
    },
    "gpt4o": {
        "cvrr_continuity_and_object_instance_count": 0.3311,
        "cvrr_fine_grained_action_understanding": 0.5441,
        "cvrr_interpretation_of_social_context": 0.6038,
        "cvrr_interpretation_of_visual_context": 0.5518,
        "cvrr_multiple_actions_in_a_single_video": 0.4424,
        "cvrr_non_existent_actions_with_existent_scene_depictions": 0.3509,
        "cvrr_non_existent_actions_with_non_existent_scene_depictions": 0.2309,
        "cvrr_partial_actions": 0.4430,
        "cvrr_time_order_understanding": 0.3440,
        "cvrr_understanding_emotional_context": 0.3223,
        "cvrr_unusual_and_physically_anomalous_activities": 0.3065
    }
}


@positional_deprecated
def simple_evaluate(
    model,
    model_args=None,
    tasks=[],
    num_fewshot=None,
    batch_size=None,
    device=None,
    limit=None,
    bootstrap_iters: int = 100000,
    check_integrity: bool = False,
    show_task_to_terminal: bool = False,
    log_samples: bool = True,
    gen_kwargs: str = None,
    cli_args=None,  # Bo: put args into more functions (cost 48 Bytes per call)
    predict_only: bool = False,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param model: Union[str, LMM]
        Name of model or LMM object, see lmms_eval.models.get_model
    :param model_args: Optional[str]
        String arguments for each model class, see LMM.create_from_arg_string.
        Ignored if `model` argument is a LMM object.
    :param tasks: list[Union[str, Task]]
        List of task names or Task objects. Task objects will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param num_fewshot: int
        Number of examples in few-shot context
    :param batch_size: int or str, optional
        Batch size for model
    :param device: str, optional
        PyTorch device (e.g. "cpu" or "cuda:0") for running models
    :param limit: int or float, optional
        Limit the number of examples per task (only use this for testing), If <1, limit is a percentage of the total number of examples.
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param check_integrity: bool
        Whether to run the relevant part of the test suite for the tasks
    :param show_task_to_terminal: bool
        If True, write out an example document and model input for checking task integrity
    :param log_samples: bool
        If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis
    :param gen_kwargs: str
        String arguments for model generation
        Ignored for all tasks with loglikelihood output_type
    :return
        Dictionary of results
    """
    random.seed(0)
    np.random.seed(1234)
    torch.manual_seed(1234)  # TODO: this may affect training runs that are run with evaluation mid-run.

    assert tasks != [], "No tasks specified, or no tasks found. Please verify the task names."

    if gen_kwargs:
        gen_kwargs = simple_parse_args_string(gen_kwargs)
        eval_logger.warning(f"generation_kwargs specified through cli, these settings will be used over set parameters in yaml tasks.")
        if gen_kwargs == "":
            gen_kwargs = None

    if model_args is None:
        model_args = ""
    lm = lmms_eval.api.registry.get_model(model).create_from_arg_string(
        model_args,
        {
            "batch_size": batch_size,
            "device": device,
        },
    )

    task_dict = lmms_eval.tasks.get_task_dict(tasks, model_name=model)
    for task_name in task_dict.keys():
        task_obj = task_dict[task_name]
        if type(task_obj) == tuple:
            group, task_obj = task_obj
            if task_obj is None:
                continue
        lm.task_dict[task_name] = task_obj.dataset

        config = task_obj._config
        if config["output_type"] == "generate_until" and gen_kwargs:
            config["generation_kwargs"].update(gen_kwargs)

        if predict_only:
            log_samples = True
            eval_logger.info(f"Processing {task_name} in output-only mode. Metrics will not be calculated!")
            # we have to change the class properties post-hoc. This is pretty hacky.
            task_obj.override_metric(metric_name="bypass")

        if num_fewshot is not None:
            if config["num_fewshot"] == 0:
                eval_logger.info(f"num_fewshot has been set to 0 for {task_name} in its config. Manual configuration will be ignored.")
            else:
                default_num_fewshot = config["num_fewshot"]
                eval_logger.warning(f"Overwriting default num_fewshot of {task_name} from {default_num_fewshot} to {num_fewshot}")

                task_obj._config["num_fewshot"] = num_fewshot

    if check_integrity:
        run_task_tests(task_list=tasks)

    results = evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=limit,
        bootstrap_iters=bootstrap_iters,
        show_task_to_terminal=show_task_to_terminal,
        log_samples=log_samples,
        cli_args=cli_args,
    )

    if lm.rank == 0:
        # add info about the model and few shot config
        results["model_configs"] = {
            "model": model if isinstance(model, str) else model.model.config._name_or_path,
            "model_args": model_args,
            "batch_size": batch_size,
            "device": device,
            "limit": limit,
            "bootstrap_iters": bootstrap_iters,
            "gen_kwargs": gen_kwargs,
        }
        # results["git_hash"] = get_git_commit_hash()
        return results
    else:
        return None


decontaminate_suffix = "_decontaminate"


@positional_deprecated
def evaluate(
    lm,
    task_dict,
    limit=None,
    bootstrap_iters: int = 100000,
    show_task_to_terminal: bool = False,
    log_samples: bool = True,
    cli_args=None,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param lm: obj
        Language Model
    :param task_dict: dict[str, Task]
        Dictionary of tasks. Tasks will be taken to have name type(task).config.task .
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param show_task_to_terminal: bool
        If True, write out an example document and model input for checking task integrity
    :param log_samples: bool
        If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis
    :return
        Dictionary of results
    """

    # stores the final result for each task, for each metric/filter pair.
    results = collections.defaultdict(dict)
    # Tracks each task's version.
    versions = collections.defaultdict(dict)
    # Tracks the YAML configs of all chosen tasks.
    configs = collections.defaultdict(dict)
    # logs info about each document evaluated.
    samples = collections.defaultdict(list)
    # tracks all Instances/requests a model must generate output on.
    requests = collections.defaultdict(list)
    # Aggregated task scores presented with groups
    results_agg = collections.defaultdict(dict)
    # Aggregated groups scores only
    groups_agg = collections.defaultdict(dict)
    # stores the amount to pad out reqs per req. type so that
    # number of fwd passes per distributed rank is equal
    padding_requests = collections.defaultdict(int)
    # store the hierarchy to do proper ordering
    task_hierarchy = collections.defaultdict(list)
    # store the ordering of tasks and groups
    task_order = collections.defaultdict(int)
    task_group_alias = collections.defaultdict(dict)
    # store num-fewshot value per task
    num_fewshot = collections.defaultdict(int)

    # get lists of each type of request
    for task_name, task in task_dict.items():
        if type(task) == tuple:
            group_name, task = task
            task_hierarchy[group_name].append(task_name)
            versions[group_name] = "N/A"

        else:
            group_name = None
            task_hierarchy[task_name] = []

        if task is None:
            continue

        versions[task_name] = task.VERSION
        configs[task_name] = dict(task.dump_config())

        if "num_fewshot" in configs[task_name]:
            n_shot = configs[task_name]["num_fewshot"]
        else:
            n_shot = 0
        num_fewshot[task_name] = n_shot

        if "task_alias" in configs[task_name]:
            task_group_alias[task_name] = configs[task_name]["task_alias"]

        if ("group_alias" in configs[task_name]) and (group_name not in task_group_alias) and (group_name is not None):
            task_group_alias[group_name] = configs[task_name]["group_alias"]

        if limit is not None:
            if task.has_test_docs():
                task_docs = task.test_docs()
            elif task.has_validation_docs():
                task_docs = task.validation_docs()
            else:
                raise RuntimeError("Task has neither test_docs nor validation_docs")
            limit = int(len(task_docs) * limit) if limit < 1.0 else int(limit)

        task.build_all_requests(limit=limit, rank=lm.rank, world_size=lm.world_size)

        eval_logger.debug(f"Task: {task_name}; number of requests on rank {lm.rank}: {len(task.instances)}")

        if show_task_to_terminal:
            for inst in task.instances:
                # print the prompt for the first few documents
                if inst.doc_id < 1:
                    eval_logger.info(
                        f"Task: {task_name}; document {inst.doc_id}; context prompt (starting on next line):\
\n{inst.args[0]}\n(end of prompt on previous line)\ntarget string or answer choice index (starting on next line):\n{task.doc_to_target(inst.doc)}\n(end of target on previous line)"
                    )
                    eval_logger.info(f"Request: {str(inst)}")

        # aggregate Instances by LMM method requested to get output.
        if cli_args.task_mode:
            # rebutal_logs_phase1_processed for rebutal
            with open('/work/LAS/wzhang-lab/mingl/code/vllm/lmms_eval_video/logs_phase1_processed/batch.json') as f:
                phase1_data = json.load(f)
            if cli_args.task_mode == 'score_reference':
                phase2_data_path = '/work/LAS/wzhang-lab/mingl/code/vllm/lmms_eval_video/logs_phase2_processed/batch.json'
                if cli_args.mixture_expert:
                    phase2_data_path = '/work/LAS/wzhang-lab/mingl/code/vllm/lmms_eval_video/logs_phase2_extra_processed/batch.json'
                with open(phase2_data_path) as f:
                    phase2_data = json.load(f)
                phase2_data = phase2_data[task.config.task]
            phase1_data = phase1_data[task.config.task]
            task_name = task.config.task
            cloned_reqs_judge = []
            prompt_dict = get_prompt()
            for index, item in enumerate(task.instances):
                for candidate_model, answer in phase1_data[str(index)].items():
                    if candidate_model == 'target':
                        continue
                    new_item = copy.deepcopy(item)
                    new_item.candidate_model = candidate_model

                    new_arguments = [item for item in new_item.arguments]
                    question = new_arguments[0]
                    if cli_args.task_mode == 'score_reference':
                        phase2_data_item = phase2_data[str(index)]
                        expert_threshold = 0.1
                        reference = [val[candidate_model] for key, val in phase2_data_item.items() if key != 'target' and key in cli_args.mixture_expert and cvrr_scores_dict[key][task_name] > expert_threshold]
                        question_new = construct_input(prompt_dict, cli_args.task_mode, reference, question, responses=answer)
                    else:
                        question_new = construct_input(prompt_dict, cli_args.task_mode, None, question, responses=answer)
                    question_new += "\nYour reply should be in JSON format: {{'rating': 'your rating number', 'reason': 'your reasoning'}}'"

                    new_arguments[0] = question_new
                    new_item.arguments = tuple(new_arguments)
                    cloned_reqs_judge.append(new_item)
            task._instances = cloned_reqs_judge

        for instance in task.instances:
            reqtype = instance.request_type
            requests[reqtype].append(instance)

        if lm.world_size > 1:
            instances_rnk = torch.tensor(len(task._instances), device=lm.device)
            gathered_item = lm.accelerator.gather(instances_rnk).cpu().detach().numpy().tolist()

            # compute number of pseudobatches to pad with (FSDP/DDP require even batches among ranks)
            numpad = max(gathered_item) - gathered_item[lm.rank]
            padding_requests[task.OUTPUT_TYPE] += numpad

    ### Run LMM on inputs, get all outputs ###
    # execute each type of request
    for reqtype, reqs in requests.items():
        eval_logger.info("Running {} requests".format(reqtype))
        # create `K` copies of each request `req` based off `K = req.repeats`
        cloned_reqs = []
        for req in reqs:
            cloned_reqs.extend([req] * req.repeats)

        if (lm.world_size > 1) and (padding_requests[reqtype] > 0):
            for _ in range(padding_requests[reqtype]):
                cloned_reqs.extend([req] * req.repeats)
        resps = getattr(lm, reqtype)(cloned_reqs)  # Choiszt run generate until    
        # put responses from model into a list of length K for each request.
        for x, req in zip(resps, cloned_reqs):
            req.resps.append(x)

        if lm.world_size > 1:
            lm.accelerator.wait_for_everyone()

    ### Postprocess outputs ###
    # TODO: del model here, maybe (idea: allow user to specify device of e.g. reward model separately)
    for task_name, task in task_dict.items():
        if type(task) == tuple:
            group, task = task
            if task is None:
                continue
        task.apply_filters()

    ### Collect values of metrics on all datapoints ###
    vals = collections.defaultdict(list)

    # unpack results and sort back in order and return control to Task
    for task_name, task in task_dict.items():
        if type(task) == tuple:
            group, task = task
            if task is None:
                continue
        # TODO: make it possible to use a different metric per filter
        # iterate over different filters used
        for key in task.instances[0].filtered_resps.keys():
            # hack: remove image columns to speed avoid loading images and speed up postprocessing
            # reason: doc_iterator will actually load image if it's in the doc.
            docs = task.test_docs() if task.has_test_docs() else task.validation_docs()
            if "d170" not in task_name and "dc100" not in task_name and "dc200" not in task_name and "llava_wilder" not in task_name and "livebench" not in task_name and "wildvision" not in task_name:
                remove_cols = []
                features = docs.features
                # If it is an Image instance or a Sequence of Image instance. Remove it
                for feature in features:
                    if isinstance(features[feature], Image):
                        remove_cols.append(feature)
                    elif isinstance(features[feature], Sequence) and isinstance(features[feature].feature, Image):
                        remove_cols.append(feature)
                if remove_cols:
                    docs = docs.remove_columns(remove_cols)

            ####################### Processing with Full Docs Mode #######################
            if task_name in ["videochatgpt_consistency"]:
                full_docs = True
            else:
                full_docs = False

            doc_iterator = itertools.islice(enumerate(docs), lm.rank, limit, lm.world_size)
            # Instead of converting the iterator to a list, use `itertools.tee` to create a parallel iterator for counting
            # doc_iterator, doc_iterator_for_counting = itertools.tee(doc_iterator)
            # Don't use above one, this would crash if doc_iterator_for_counting contains too many objects and very slow
            doc_iterator_for_counting = itertools.islice(range(len(task.test_docs())), lm.rank, limit, lm.world_size) if task.has_test_docs() else itertools.islice(range(len(task.validation_docs())), lm.rank, limit, lm.world_size)
            total_docs = sum(1 for _ in doc_iterator_for_counting)
            pbar = tqdm(total=total_docs, desc=f"Postprocessing", disable=(lm.rank != 0))
            for doc_id, doc in doc_iterator:
                # subset instances to only this document id ; sort by idx
                requests = list(filter(lambda x: x.doc_id == doc_id, task.instances))
                requests.sort(key=lambda x: x.idx)
                if cli_args.task_mode:
                    metrics = {}
                else:
                    if full_docs:
                        metrics = task.process_results(doc, [req.filtered_resps[key] for req in requests], full_docs=docs)
                    else:
                        metrics = task.process_results(doc, [req.filtered_resps[key] for req in requests])
                if log_samples:
                    target = task.doc_to_target(doc)
                    if cli_args.task_mode:
                        example = {
                            "doc_id": doc_id,
                            "target": target,
                            "doc": doc,
                            "arguments": [tuple(a for a in req.args if isinstance(a, (int, str))) for req in requests],  # do not include image
                            "resps": [req.resps for req in requests],
                            "filtered_resps": [req.filtered_resps[key] for req in requests],
                            "candidate_model": [req.candidate_model for req in requests],
                        }
                    else:
                        example = {
                            "doc_id": doc_id,
                            "target": target,
                            "doc": doc,
                            "arguments": [tuple(a for a in req.args if isinstance(a, (int, str))) for req in requests],  # do not include image
                            "resps": [req.resps for req in requests],
                            "filtered_resps": [req.filtered_resps[key] for req in requests],
                        }

                    example.update(metrics)
                    samples[task_name].append(example)
                    # with open('/work/LAS/wzhang-lab/mingl/code/vllm/lmms_eval_video/videl_llava_log.txt', 'a') as f:
                        # json.dump(example, f)
                        # f.write('\n')
                for metric, value in metrics.items():
                    vals[(task_name, key, metric)].append(value)
                pbar.update(1)

            pbar.close()

    if lm.world_size > 1:
        # if multigpu, then gather data across all ranks
        # first gather logged samples across all ranks
        for task_name, task_samples in list(samples.items()):
            full_samples = [None] * lm.world_size
            torch.distributed.all_gather_object(full_samples, task_samples)
            samples[task_name] = list(itertools.chain.from_iterable(full_samples))
        # then collect metrics across all ranks
        vals_torch = collections.defaultdict(list)
        for (task_name, key, metric), items in vals.items():
            numitem = 0
            if type(items[0]) == tuple:
                numitem = len(items[0])

            if isinstance(items[0], (str, list, dict)):
                # handle the string case
                gathered_items = [None] * lm.accelerator.num_processes
                torch.distributed.all_gather_object(gathered_items, items)

                gathered_item = list(itertools.chain.from_iterable(gathered_items))
            else:
                # distributed gather requires all ranks to have same dimensions
                # so we pad out with float32 min value
                pad_value = torch.finfo(torch.float32).min
                metrics_tensor = torch.tensor(items, device=lm.device)

                original_dtype = metrics_tensor.dtype  # store original dtype
                torch_device_tensor = lm.accelerator.pad_across_processes(metrics_tensor.to(torch.float32), pad_index=pad_value)
                gathered_item = lm.accelerator.gather(torch_device_tensor)

                if numitem > 0:
                    gathered_filtered = gathered_item[gathered_item[:, 0] != pad_value]
                else:
                    gathered_filtered = gathered_item[gathered_item != pad_value]

                gathered_item = gathered_filtered.to(original_dtype).cpu().detach().numpy().tolist()
                # reconvert if we were passed a tuple of values
                if numitem > 0:
                    gathered_item = [tuple(g) for g in gathered_item]

            if lm.rank == 0:
                vals_torch[(task_name, key, metric)] = gathered_item

        vals = vals_torch
        # Ensure all ranks wait for rank 0 to finish aggregation
        torch.distributed.barrier()

    # Synchronize processes with a temp file in case the evluation metric requires gpus
    # TODO: fix barriers' taking up gpu computation
    os.makedirs(cli_args.output_path, exist_ok=True)
    if os.path.exists(f"{cli_args.output_path}/rank{int(os.environ.get('RANK', 0))}_metric_eval_done.txt"):
        os.remove(f"{cli_args.output_path}/rank{int(os.environ.get('RANK', 0))}_metric_eval_done.txt")

    if lm.rank == 0:
        ### Get task ordering for correct sample-wide aggregation
        group_to_task = {}
        for group in task_hierarchy.keys():
            if group not in task_order:
                task_order[group] = 0

            if len(task_hierarchy[group]) > 0:
                group_to_task[group] = task_hierarchy[group].copy()

            for task in task_hierarchy[group]:
                if task in task_order:
                    task_order[task] += 1
                else:
                    task_order[task] = 1 + task_order[group]

                if task in task_hierarchy:
                    group_to_task[group].remove(task)
                    group_to_task[group].extend(task_hierarchy[task])

        task_to_group = {}
        for group in group_to_task:
            for task in group_to_task[group]:
                if task in task_to_group:
                    task_to_group[task].append(group)
                else:
                    task_to_group[task] = [group]

        ### Aggregate results over all datapoints ###
        # aggregate results ; run bootstrap CIs
        for (task_name, key, metric), items in vals.items():
            task = task_dict[task_name]
            metric_key = metric + "," + key

            if type(task) == tuple:
                group_name, task = task
            else:
                group_name = None

            if metric not in task.aggregation():
                continue

            agg_fn = task.aggregation()[metric]

            # Bo: for models that need to know the args to save to correct path
            if inspect.getfullargspec(agg_fn).args == ["results", "args"]:
                results[task_name][metric_key] = agg_fn(items, cli_args)
            else:
                # Bo: for models only need agg items
                results[task_name][metric_key] = agg_fn(items)

            results[task_name]["samples"] = len(items)

            # hotfix: bleu, chrf, ter seem to be really expensive to bootstrap
            # so we run them less iterations. still looking for a cleaner way to do this
            if bootstrap_iters > 0:
                stderr = lmms_eval.api.metrics.stderr_for_metric(
                    metric=task.aggregation()[metric],
                    bootstrap_iters=min(bootstrap_iters, 100) if metric in ["bleu", "chrf", "ter"] else bootstrap_iters,
                )

                if stderr is not None and len(items) > 1:
                    results[task_name][metric + "_stderr" + "," + key] = stderr(items)
                else:
                    results[task_name][metric + "_stderr" + "," + key] = "N/A"

        if bool(results):
            for group, task_list in reversed(task_hierarchy.items()):
                if task_list == []:
                    total_size = results[group]["samples"]
                else:
                    total_size = 0

                    for task in task_list:
                        metrics = results[task]

                        current_size = metrics.pop("samples")
                        # TODO: There should be a way for users
                        #       to toggle between weighted and
                        #       unweighted averaging
                        # For unweighted averaging, use:
                        #     current_size = 1

                        all_stderr = []
                        for metric in [key for key in metrics.keys() if "_stderr" not in key]:
                            stderr = "_stderr,".join(metric.split(","))
                            stderr_score = results[task][stderr]
                            var_score = stderr_score**2 if stderr_score != "N/A" else 0
                            metric_score = results[task][metric]

                            all_stderr.append(stderr)

                            if metric_score is None:
                                results[group][metric] = None
                                results[group][stderr] = 0
                                continue

                            if metric in results[group]:
                                if isinstance(results[group][metric], str) == False:
                                    results[group][metric] = (results[group][metric] * total_size + metric_score * current_size) / (total_size + current_size)
                                    # $$s_z^2 = \frac{(n-1) s_x^2 + (m-1) s_y^2}{n+m-1} + \frac{nm(\bar x - \bar y)^2}{(n+m)(n+m-1)}.$$
                                    results[group][stderr] = ((total_size - 1) * results[group][stderr] + (current_size - 1) * var_score) / (total_size + current_size - 1) + total_size * current_size / (
                                        (total_size + current_size) * (total_size + current_size - 1)
                                    ) * (results[group][metric] - metric_score) ** 2
                                else:
                                    # accuracy = re.search(r'acc: ([\d.]+)%', results[group][metric]).group(1)
                                    # score = re.search(r'score: ([\d.]+)', results[group][metric]).group(1)
                                    # group_accuracy = float(accuracy)
                                    # group_score = float(score)
                                    # group_accuracy = (group_accuracy * total_size + metric_score * current_size) / total_size
                                    # group_score = (group_score * total_size + metric_score * current_size) / total_size
                                    # results[group][metric] = "Acc: " + str(group_accuracy) + " Score: " + str(group_score)
                                    results[group][metric] = "group_results"
                                    results[group][stderr] = 0
                            else:
                                results[group][metric] = metric_score
                                results[group][stderr] = var_score

                        total_size += current_size

                    for stderr in all_stderr:
                        results[group][stderr] = np.sqrt(results[group][stderr])

                results[group]["samples"] = total_size

        def print_tasks(task_hierarchy, task_order, task_version, task_group_alias):
            results_agg = collections.defaultdict(dict)
            groups_agg = collections.defaultdict(dict)
            for group_name, task_list in task_hierarchy.items():
                order = task_order[group_name]
                results_agg[group_name] = results[group_name].copy()
                results_agg[group_name]["tab"] = order

                if (order < max(task_order.values())) and (len(task_list) > 0):
                    groups_agg[group_name] = results[group_name].copy()
                    groups_agg[group_name]["tab"] = order

                if task_list != []:
                    for task in sorted(task_list):
                        if task in task_hierarchy:
                            _task_hierarchy = {task: task_hierarchy[task]}
                        else:
                            _task_hierarchy = {task: []}

                        _results_agg, _groups_agg, task_version = print_tasks(_task_hierarchy, task_order, task_version, task_group_alias)

                        results_agg = {**results_agg, **_results_agg}
                        groups_agg = {**groups_agg, **_groups_agg}

            return results_agg, groups_agg, task_version

        results_agg, groups_agg, versions = print_tasks(task_hierarchy, task_order, versions, task_group_alias)

        for task in results_agg:
            task_results = results_agg[task]

            if "samples" in task_results:
                task_results.pop("samples")

            tab_string = ""
            if "tab" in task_results:
                tab = task_results.pop("tab")
                tab_string = " " * tab + "- " if tab > 0 else ""

            if task in task_group_alias:
                task_alias = task_group_alias[task]
                results_agg[task]["alias"] = tab_string + task_alias
            else:
                results_agg[task]["alias"] = tab_string + task

        for group in groups_agg:
            group_results = groups_agg[group]

            if "samples" in group_results:
                group_results.pop("samples")

            tab_string = ""
            if "tab" in group_results:
                tab = group_results.pop("tab")
                tab_string = " " * tab + "- " if tab > 0 else ""

            if group in task_group_alias:
                group_alias = task_group_alias[group]
                groups_agg[group]["alias"] = tab_string + group_alias
            else:
                groups_agg[group]["alias"] = tab_string + group

        for group_name, task_list in task_hierarchy.items():
            if task_list != []:
                num_fewshot[group_name] = num_fewshot[task_list[0]]

        results_dict = {
            "results": dict(results_agg.items()),
            **({"groups": dict(groups_agg.items())} if bool(groups_agg) else {}),
            "configs": dict(sorted(configs.items())),
            "versions": dict(sorted(versions.items())),
            "n-shot": dict(sorted(num_fewshot.items())),
        }
        if log_samples:
            results_dict["samples"] = dict(samples)
    else:
        results_dict = None
    
    with open(f"{cli_args.output_path}/rank{int(os.environ.get('RANK', 0))}_metric_eval_done.txt", 'w') as f:
        f.write(f"rank {int(os.environ.get('RANK', 0))} eval done")
    # while len([file for file in os.listdir(cli_args.output_path) if file.endswith('metric_eval_done.txt')]) < lm.accelerator.num_processes:
    #     time.sleep(1)

    return results_dict
 