#!/bin/bash

# List of target folders
folders=(
    "cvrr_continuity_and_object_instance_count"
    "cvrr_fine_grained_action_understanding"
    "cvrr_interpretation_of_social_context"
    "cvrr_interpretation_of_visual_context"
    "cvrr_multiple_actions_in_a_single_video"
    "cvrr_non_existent_actions_with_existent_scene_depictions"
    "cvrr_non_existent_actions_with_non_existent_scene_depictions"
    "cvrr_partial_actions"
    "cvrr_time_order_understanding"
    "cvrr_understanding_emotional_context"
    "cvrr_unusual_and_physically_anomalous_activities"
)

# Create each target folder if it doesn't exist
for folder in "${folders[@]}"; do
    mkdir -p "$folder"
done

# Move folders to their respective target folders based on the pattern in the folder name
for dir in */; do
    if [[ -d $dir ]]; then
        if [[ $dir == *"cvrr_continuity_and_object_instance_count"* ]]; then
            mv "$dir" "cvrr_continuity_and_object_instance_count/"
        elif [[ $dir == *"cvrr_fine_grained_action_understanding"* ]]; then
            mv "$dir" "cvrr_fine_grained_action_understanding/"
        elif [[ $dir == *"cvrr_interpretation_of_social_context"* ]]; then
            mv "$dir" "cvrr_interpretation_of_social_context/"
        elif [[ $dir == *"cvrr_interpretation_of_visual_context"* ]]; then
            mv "$dir" "cvrr_interpretation_of_visual_context/"
        elif [[ $dir == *"cvrr_multiple_actions_in_a_single_video"* ]]; then
            mv "$dir" "cvrr_multiple_actions_in_a_single_video/"
        elif [[ $dir == *"cvrr_non_existent_actions_with_existent_scene_depictions"* ]]; then
            mv "$dir" "cvrr_non_existent_actions_with_existent_scene_depictions/"
        elif [[ $dir == *"cvrr_non_existent_actions_with_non_existent_scene_depictions"* ]]; then
            mv "$dir" "cvrr_non_existent_actions_with_non_existent_scene_depictions/"
        elif [[ $dir == *"cvrr_partial_actions"* ]]; then
            mv "$dir" "cvrr_partial_actions/"
        elif [[ $dir == *"cvrr_time_order_understanding"* ]]; then
            mv "$dir" "cvrr_time_order_understanding/"
        elif [[ $dir == *"cvrr_understanding_emotional_context"* ]]; then
            mv "$dir" "cvrr_understanding_emotional_context/"
        elif [[ $dir == *"cvrr_unusual_and_physically_anomalous_activities"* ]]; then
            mv "$dir" "cvrr_unusual_and_physically_anomalous_activities/"
        fi
    fi
done

echo "Folders moved successfully."


#!/bin/bash

# List of target folders
folders=(
    "cvrr_continuity_and_object_instance_count"
    "cvrr_fine_grained_action_understanding"
    "cvrr_interpretation_of_social_context"
    "cvrr_interpretation_of_visual_context"
    "cvrr_multiple_actions_in_a_single_video"
    "cvrr_non_existent_actions_with_existent_scene_depictions"
    "cvrr_non_existent_actions_with_non_existent_scene_depictions"
    "cvrr_partial_actions"
    "cvrr_time_order_understanding"
    "cvrr_understanding_emotional_context"
    "cvrr_unusual_and_physically_anomalous_activities"
)

# Loop through each target folder
for folder in "${folders[@]}"; do
    # Find and remove the specified files in each subfolder
    find "$folder" -type f \( -name "rank0_metric_eval_done.txt" -o -name "results.json" \) -exec rm {} +
done

echo "Specified files removed successfully."


#!/bin/bash

# List of target folders
folders=(
    "cvrr_continuity_and_object_instance_count"
    "cvrr_fine_grained_action_understanding"
    "cvrr_interpretation_of_social_context"
    "cvrr_interpretation_of_visual_context"
    "cvrr_multiple_actions_in_a_single_video"
    "cvrr_non_existent_actions_with_existent_scene_depictions"
    "cvrr_non_existent_actions_with_non_existent_scene_depictions"
    "cvrr_partial_actions"
    "cvrr_time_order_understanding"
    "cvrr_understanding_emotional_context"
    "cvrr_unusual_and_physically_anomalous_activities"
)

# Loop through each target folder
for folder in "${folders[@]}"; do
    # Loop through each subfolder within the target folder
    for subfolder in "$folder"/*; do
        if [[ -d $subfolder ]]; then
            # Extract the base name of the subfolder
            base_name=$(basename "$subfolder")
            
            # Remove the first 10 characters from the base name
            new_base_name="${base_name:10}"
            
            # Construct the new subfolder path
            new_subfolder="$folder/$new_base_name"
            
            # Rename the subfolder
            mv "$subfolder" "$new_subfolder"
            
            # Remove the specified files in the renamed subfolder
            rm -f "$new_subfolder/rank0_metric_eval_done.txt"
            rm -f "$new_subfolder/results.json"
        fi
    done
done

echo "Prefixes removed and specified files deleted successfully."


# List of target folders
folders=(
    "cvrr_continuity_and_object_instance_count"
    "cvrr_fine_grained_action_understanding"
    "cvrr_interpretation_of_social_context"
    "cvrr_interpretation_of_visual_context"
    "cvrr_multiple_actions_in_a_single_video"
    "cvrr_non_existent_actions_with_existent_scene_depictions"
    "cvrr_non_existent_actions_with_non_existent_scene_depictions"
    "cvrr_partial_actions"
    "cvrr_time_order_understanding"
    "cvrr_understanding_emotional_context"
    "cvrr_unusual_and_physically_anomalous_activities"
)

# Loop through each target folder
for folder in "${folders[@]}"; do
    # Loop through each subfolder within the target folder
    for subfolder in "$folder"/*; do
        if [[ -d $subfolder ]]; then
            # Extract the base name of the subfolder
            base_name=$(basename "$subfolder")
            
            # Check if the folder is one of the last three, if so, modify the prefix
            if [[ "$folder" == "vatex_test" ]]; then
                prefix="${base_name%%_vatex_test_*}"
            elif [[ "$folder" == "videochatgpt_gen" ]]; then
                prefix="${base_name%%_videochatgpt_gen_*}"
            elif [[ "$folder" == "youcook2_val" ]]; then
                prefix="${base_name%%_youcook2_val_*}"
            else
                prefix="${base_name%%_cvrr_*}"
            fi
            
            # Construct the new subfolder path
            new_subfolder="$folder/$prefix"
            
            # Rename the subfolder
            mv "$subfolder" "$new_subfolder"
        fi
    done
done