#!/bin/bash

# List of model keys
MODEL_KEYS=(
    "video_chatgpt"
    "video_llava"
    "gpt4v"
    "mplug_owl_Video"
    "llama_vid"
)

# Array of tasks
TASKS=( 
    "videochatgpt_gen"
)

# Loop through the list of model keys
for model_key in "${MODEL_KEYS[@]}"; do
    echo "Launching model: $model_key..."
    
    # Nested loop for tasks
    for task in "${TASKS[@]}"; do
        echo "Processing task: $task..."
        
        # Check if "cvrr" is not in task
        if [[ $task != *"cvrr"* ]]; then
            accelerate launch --num_processes=16 -m lmms_eval --model $model_key --tasks $task --batch_size 1 --limit 1000 --log_samples --log_samples_suffix "${model_key}_${task}" --output_path ./logs/
        else
            accelerate launch --num_processes=16 -m lmms_eval --model $model_key --tasks $task --batch_size 1 --log_samples --log_samples_suffix "${model_key}_${task}" --output_path ./logs/
        fi
    done
done

