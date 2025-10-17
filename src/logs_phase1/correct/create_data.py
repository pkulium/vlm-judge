import os
import json

# List of target folders
folders = [
    "cvrr_continuity_and_object_instance_count",
    "cvrr_fine_grained_action_understanding",
    "cvrr_interpretation_of_social_context",
    "cvrr_interpretation_of_visual_context",
    "cvrr_multiple_actions_in_a_single_video",
    "cvrr_non_existent_actions_with_existent_scene_depictions",
    "cvrr_non_existent_actions_with_non_existent_scene_depictions",
    "cvrr_partial_actions",
    "cvrr_time_order_understanding",
    "cvrr_understanding_emotional_context",
    "cvrr_unusual_and_physically_anomalous_activities",
]

# Initialize the main dictionary to hold all the data
all_data = {}

# Loop through each target folder
for folder in folders:
    folder_data = {}
    
    # Loop through each subfolder within the target folder
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            subfolder_data = {}
            
            # Loop through each JSON file in the subfolder
            for file in os.listdir(subfolder_path):
                if file.endswith("_debatedebate.json"):
                    file_path = os.path.join(subfolder_path, file)
                    
                    # Read the JSON file
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Extract the relevant data
                    logs = data.get("logs", [])
                    log_data = {log["doc_id"]: log["filtered_resps"] for log in logs}
                    
                    # Store the extracted data in the subfolder dictionary
                    subfolder_data.update(log_data)
            
            # Store the subfolder data in the folder dictionary
            folder_data[subfolder] = subfolder_data
    
    # Store the folder data in the main dictionary
    all_data[folder] = folder_data

# Write the combined data to a single JSON file
output_file = "single.json"
with open(output_file, 'w') as f:
    json.dump(all_data, f, indent=4)

print("Combined JSON file created successfully.")


import os
import json

# List of target folders
folders = [
    "cvrr_continuity_and_object_instance_count",
    "cvrr_fine_grained_action_understanding",
    "cvrr_interpretation_of_social_context",
    "cvrr_interpretation_of_visual_context",
    "cvrr_multiple_actions_in_a_single_video",
    "cvrr_non_existent_actions_with_existent_scene_depictions",
    "cvrr_non_existent_actions_with_non_existent_scene_depictions",
    "cvrr_partial_actions",
    "cvrr_time_order_understanding",
    "cvrr_understanding_emotional_context",
    "cvrr_unusual_and_physically_anomalous_activities",
]

# Initialize the main dictionary to hold all the data
all_data = {}

# Loop through each target folder
for folder in folders:
    folder_data = {}
    
    # Loop through each subfolder within the target folder
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            # Loop through each JSON file in the subfolder
            for file in os.listdir(subfolder_path):
                if file.endswith("_debatedebate.json"):
                    file_path = os.path.join(subfolder_path, file)
                    
                    # Read the JSON file
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Extract the relevant data
                    logs = data.get("logs", [])
                    for log in logs:
                        doc_id = log["doc_id"]
                        filtered_resps = log["filtered_resps"]
                        filtered_resps.append(log['gpt_eval_score']['score'])
                        target = log["target"]
                        
                        # Initialize the doc_id dictionary if not present
                        if doc_id not in folder_data:
                            folder_data[doc_id] = {}
                        
                        if "target" not in folder_data[doc_id]:
                            folder_data[doc_id]["target"] = [target]
                        # Add the subfolder data under the doc_id
                        folder_data[doc_id][subfolder] = filtered_resps
                       
    
    # Store the folder data in the main dictionary
    all_data[folder] = folder_data

# Write the combined data to a single JSON file
output_file = "batch_with_score.json"
with open(output_file, 'w') as f:
    json.dump(all_data, f, indent=4)

print("Combined JSON file created successfully.")
