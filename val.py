import json
import random
import os

def read_jsonl_file(filepath):
    """
    Reads a JSONL file and returns a list of its lines (each line is a string).

    Args:
        filepath (str): The path to the JSONL file.

    Returns:
        list: A list of strings, where each string is a line from the file.
              Returns an empty list if the file is not found or empty.
    """
    lines = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                lines.append(line.strip()) # .strip() removes leading/trailing whitespace including newlines
        print(f"Successfully read {len(lines)} lines from '{filepath}'.")
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'. Please check the path.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
    return lines

def select_random_lines(lines, percentage):
    """
    Randomly selects a specified percentage of lines from a given list of lines.

    Args:
        lines (list): A list of lines (strings) to select from.
        percentage (float): The percentage of lines to select (e.g., 0.15 for 15%).

    Returns:
        tuple: A tuple containing two lists:
               - selected_lines: The randomly selected lines.
               - remaining_lines: The lines not selected.
    """
    if not (0 <= percentage <= 1):
        raise ValueError("Percentage must be between 0 and 1 (inclusive).")
    
    if not lines:
        return [], []

    num_to_select = int(len(lines) * percentage)
    
    # Shuffle the lines to ensure random selection
    shuffled_lines = lines[:] # Create a copy to avoid modifying the original list
    random.shuffle(shuffled_lines)

    selected_lines = shuffled_lines[:num_to_select]
    remaining_lines = shuffled_lines[num_to_select:]

    print(f"Selected {len(selected_lines)} lines ({percentage*100:.2f}%) for fine-tuning.")
    print(f"Remaining {len(remaining_lines)} lines.")
    
    return selected_lines, remaining_lines

def save_lines_to_file(lines, filepath):
    """
    Saves a list of lines to a specified file, each on a new line.

    Args:
        lines (list): A list of strings to save.
        filepath (str): The path to the output file.
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line + '\n')
        print(f"Successfully saved {len(lines)} lines to '{filepath}'.")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")

if __name__ == "__main__":
    # --- Configuration ---
    input_filename = "clinical_finetune_dataset_no_common_meds (1)_gemini_fixed.jsonl"  # Name of your input JSONL file
    selection_percentage = 0.15   # 15% for fine-tuning

    output_finetune_filename = "F_clinical.jsonl"
    output_remaining_filename = "FVAL_clinical.jsonl" # Optional: Save the rest of the data

    # --- Create a dummy JSONL file for testing if it doesn't exist ---
    if not os.path.exists(input_filename):
        print(f"'{input_filename}' not found. Creating a dummy file for demonstration.")
       


    # --- Main Workflow ---
    print(f"Starting data processing for '{input_filename}'...")
    
    # 1. Read the JSONL file
    all_lines = read_jsonl_file(input_filename)

    if all_lines:
        # 2. Select random lines
        finetune_lines, remaining_lines = select_random_lines(all_lines, selection_percentage)

        # 3. Save the selected lines for fine-tuning
        save_lines_to_file(finetune_lines, output_finetune_filename)

        # 4. Save the remaining lines (optional)
        save_lines_to_file(remaining_lines, output_remaining_filename)
        
        print("\nProcessing complete. Check the generated files:")
        print(f"- Fine-tuning data: {output_finetune_filename}")
        print(f"- Remaining data: {output_remaining_filename}")
    else:
        print("No lines were read. Exiting.")

