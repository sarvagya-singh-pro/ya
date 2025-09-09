import json

def convert_to_gemini_format(input_file_path, output_file_path):
    """
    Reads a JSON file with clinical case data and converts it into a JSONL
    format suitable for Gemini fine-tuning.

    Args:
        input_file_path (str): The path to the input JSON file.
        output_file_path (str): The path to the output JSONL file.
    """
    try:
        # Open and load the original JSON data file
        with open(input_file_path, 'r', encoding='utf-8') as f_in:
            training_data = json.load(f_in)

        # Open the output file to write the converted data in JSONL format
        with open(output_file_path, 'w', encoding='utf-8') as f_out:
            # Counter for successfully processed records
            records_processed = 0
            
            # Iterate over each record in the training data
            for record in training_data:
                # The full text of the case is in the 'output' field
                full_text = record.get("output", "")
                if not full_text:
                    print(f"Skipping a record because its 'output' field is empty: {record}")
                    continue

                # The analysis part starts with 'Clinical Question:'. We use this to split the text.
                split_keyword = "Clinical Question:"
                parts = full_text.split(f'\n\n{split_keyword}', 1)
                
                # Ensure the split was successful
                if len(parts) != 2:
                    print(f"Skipping a record because the expected format was not found: {record}")
                    continue

                # The first part contains the patient info. We extract it.
                patient_info_section = parts[0]
                patient_info_start_index = patient_info_section.find("Patient Information:")
                if patient_info_start_index == -1:
                    print(f"Skipping a record because 'Patient Information:' was not found: {record}")
                    continue
                
                patient_info_block = patient_info_section[patient_info_start_index:]

                # The second part is the clinical analysis. We add the keyword back.
                analysis_block = f"{split_keyword}{parts[1]}"

                # Construct the input prompt for the model
                input_text = (
                    "Analyze this clinical case and provide a comprehensive assessment with treatment recommendations:\n\n"
                    f"{patient_info_block}"
                )

                # The output text is the analysis part
                output_text = analysis_block
                
                # Create the new data structure for Gemini fine-tuning
                gemini_record = {
                    "input_text": input_text,
                    "output_text": output_text
                }
                
                # Write the converted record as a new line in the output file
                f_out.write(json.dumps(gemini_record) + '\n')
                records_processed += 1

        print(f"\nConversion successful!")
        print(f"Processed {records_processed} records.")
        print(f"Output saved to: {output_file_path}")

    except FileNotFoundError:
        print(f"Error: The file '{input_file_path}' was not found.")
        print("Please make sure the file exists and is in the same directory as the script.")
    except json.JSONDecodeError:
        print(f"Error: The file '{input_file_path}' is not a valid JSON file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    # Define the input and output file names
    input_filename = 'clinical_llm_training_data.json'
    output_filename = 'converted_tuning_data.jsonl'
    
    # Run the conversion function
    convert_to_gemini_format(input_filename, output_filename)