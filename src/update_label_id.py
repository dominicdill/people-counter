import os
from pprint import pprint
from collections import defaultdict

def update_txt_files(target_int, directory_path):
    """
    Recursively process all .txt files in the given directory and its subdirectories.
    
    For each file, only keep lines where the first space-delimited piece equals `target_int`.
    In those lines, the first token is replaced with 0.
    
    Parameters:
        target_int (int): The integer to match against the first token in each line.
        directory_path (str): The root directory to start processing .txt files.
    """
    log_dict = dict()
    files_changed_dict = defaultdict(list)
    for root, _, files in os.walk(directory_path):
        log_dict[root] = 0
        for file in files:
            if file.lower().endswith('.txt'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                except Exception as e:
                    print(f"Could not read {file_path}: {e}")
                    continue

                new_lines = []
                for line in lines:
                    # Remove leading/trailing whitespace
                    stripped_line = line.strip()
                    # Skip blank lines
                    if not stripped_line:
                        continue

                    parts = stripped_line.split()
                    
                    # Check if the first token is an integer and equals target_int
                    try:
                        if int(parts[0]) == target_int:
                            # Replace the first token with "0"
                            parts[0] = "0"
                            # Reconstruct the line and add a newline character
                            new_lines.append(" ".join(parts) + "\n")
                    except ValueError:
                        # If the first token isn't an integer, skip the line.
                        continue
                
                # Write the updated content back to the file.
                try:
                    with open(file_path, 'w') as f:
                        f.writelines(new_lines)
                        if new_lines != lines:
                            log_dict[root] += 1
                            files_changed_dict[root].append(file)

                except Exception as e:
                    print(f"Could not write to {file_path}: {e}")
    print_flag = input("Do you want to see how many files were changed in each directory? (y/n): ")
    if print_flag.lower() == 'y':
        pprint(log_dict)
    print_flag = input("Do you want to see the list of files changed in each directory? (y/n): ")
    if print_flag.lower() == 'y':
        pprint(files_changed_dict)

if __name__ == "__main__":
    target_int = int(input("Enter the target integer: "))
    directory_path = input("Enter the directory path: ")
    update_txt_files(target_int, directory_path)