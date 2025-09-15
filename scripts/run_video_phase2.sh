#!/bin/bash

# List of model keys
MODEL_KEYS=(
    "video_llava"
    "video_chatgpt"
    "gpt4v"
    "internvl2"
    "llama_vid"
    "mplug_owl_Video"
)

# Array of tasks
TASKS=( 
     "cvrr_time_order_understanding"
     "cvrr_continuity_and_object_instance_count"
     "cvrr_fine_grained_action_understanding"
     "cvrr_interpretation_of_social_context"
     "cvrr_interpretation_of_visual_context"
     "cvrr_multiple_actions_in_a_single_video"
     "cvrr_non_existent_actions_with_existent_scene_depictions"
     "cvrr_non_existent_actions_with_non_existent_scene_depictions"
     "cvrr_partial_actions"
     "cvrr_understanding_emotional_context"
     "cvrr_unusual_and_physically_anomalous_activities"
)

# Loop through the list of model keys
for model_key in "${MODEL_KEYS[@]}"; do
    echo "Launching model: $model_key..."
    
    # Nested loop for tasks
    for task in "${TASKS[@]}"; do
        echo "Processing task: $task..."
        
        # Check if "cvrr" is not in task
        if [[ $task != *"cvrr"* ]]; then
            accelerate launch --num_processes=32 -m lmms_eval --model $model_key --tasks $task --batch_size 1 --limit 1000 --log_samples --log_samples_suffix "${model_key}_${task}" --task_mode score --output_path ./logs_p2/
        else
            accelerate launch --num_processes=32 -m lmms_eval --model $model_key --tasks $task --batch_size 1 --log_samples --log_samples_suffix "${model_key}_${task}" --task_mode score --output_path ./logs_p2/
        fi
    done
done

