import re
import os

def update_paths_in_script(script_path, old_base_path, new_base_path):
    with open(script_path, 'r') as file:
        script_content = file.read()

    updated_content = re.sub(rf'("{old_base_path}[^"]*")', lambda match: match.group(1).replace(old_base_path, new_base_path), script_content)

    with open(script_path, 'w') as file:
        file.write(updated_content)

def update_all_paths_in_project(project_directory, old_base_path, new_base_path):
    for root, _, files in os.walk(project_directory):
        for file in files:
            if file.endswith('.py'):
                script_path = os.path.join(root, file)
                update_paths_in_script(script_path, old_base_path, new_base_path)

if __name__ == "__main__":
    project_directory = '/home/jman/tmp/test2.1'
    old_base_path = 'training_data'
    new_base_path = 'training_data_v2'
    update_all_paths_in_project(project_directory, old_base_path, new_base_path)
