import json
import os
from typing import List, Dict, Any, Optional

def validate_gemini_finetune_format(input_file: str, output_file: str = None) -> bool:
    """
    Validate and fix JSONL file for Gemini fine-tuning format.
    
    Required format for each line:
    {
        "contents": [
            {"role": "user", "parts": [{"text": "..."}]},
            {"role": "model", "parts": [{"text": "..."}]}
        ]
    }
    
    Args:
        input_file (str): Path to input JSONL file
        output_file (str): Path to output fixed file
        
    Returns:
        bool: True if successful
    """
    
    if output_file is None:
        name, ext = os.path.splitext(input_file)
        output_file = f"{name}_gemini_fixed{ext}"
    
    valid_records = []
    invalid_records = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                    
                    # Fix the record to match Gemini format
                    fixed_record = fix_gemini_format(record, line_num)
                    
                    if fixed_record:
                        # Validate the fixed record
                        if validate_single_record(fixed_record):
                            valid_records.append(fixed_record)
                        else:
                            invalid_records.append((line_num, record, "Invalid after fixing"))
                    else:
                        invalid_records.append((line_num, record, "Could not fix format"))
                        
                except json.JSONDecodeError as e:
                    invalid_records.append((line_num, line, f"JSON decode error: {e}"))
        
        # Write valid records to output file
        with open(output_file, 'w', encoding='utf-8') as file:
            for record in valid_records:
                file.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        print(f"=== GEMINI FINE-TUNING FORMAT VALIDATION ===")
        print(f"Total records processed: {len(valid_records) + len(invalid_records)}")
        print(f"Valid records: {len(valid_records)}")
        print(f"Invalid records: {len(invalid_records)}")
        print(f"Fixed file saved as: {output_file}")
        
        if invalid_records:
            print(f"\n=== INVALID RECORDS (first 5) ===")
            for line_num, content, error in invalid_records[:5]:
                print(f"Line {line_num}: {error}")
                if isinstance(content, str):
                    print(f"Content: {content[:100]}...")
                else:
                    print(f"Content keys: {list(content.keys()) if isinstance(content, dict) else 'Not a dict'}")
                print("-" * 50)
        
        return len(invalid_records) == 0
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        return False
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

def fix_gemini_format(record: Dict[str, Any], line_num: int) -> Optional[Dict[str, Any]]:
    """
    Fix a record to match Gemini fine-tuning format.
    
    Args:
        record (dict): Original record
        line_num (int): Line number for debugging
        
    Returns:
        dict: Fixed record or None if unfixable
    """
    
    try:
        # Case 1: Record already has correct format
        if "contents" in record and isinstance(record["contents"], list):
            return record
        
        # Case 2: Record has systemInstruction and contents (your current format)
        if "systemInstruction" in record and "contents" in record:
            # Convert to Gemini format
            contents = []
            
            # Add system instruction as first message if present
            if record["systemInstruction"] and "parts" in record["systemInstruction"]:
                system_text = record["systemInstruction"]["parts"][0]["text"]
                # You might want to prepend this to the user message instead
                # For now, we'll skip system instruction as Gemini format doesn't directly support it
            
            # Add the conversation contents
            if isinstance(record["contents"], list):
                contents.extend(record["contents"])
            
            return {"contents": contents}
        
        # Case 3: Record has individual messages that need to be structured
        if "role" in record and "parts" in record:
            # Single message, convert to conversation format
            return {"contents": [record]}
        
        # Case 4: Try to extract conversation from other formats
        if "messages" in record:
            contents = []
            for msg in record["messages"]:
                if "role" in msg and "content" in msg:
                    # Convert OpenAI format to Gemini format
                    role = "model" if msg["role"] == "assistant" else msg["role"]
                    contents.append({
                        "role": role,
                        "parts": [{"text": msg["content"]}]
                    })
            return {"contents": contents}
        
        # Case 5: Try to extract from other common formats
        if "prompt" in record and "response" in record:
            return {
                "contents": [
                    {"role": "user", "parts": [{"text": record["prompt"]}]},
                    {"role": "model", "parts": [{"text": record["response"]}]}
                ]
            }
        
        if "question" in record and "answer" in record:
            return {
                "contents": [
                    {"role": "user", "parts": [{"text": record["question"]}]},
                    {"role": "model", "parts": [{"text": record["answer"]}]}
                ]
            }
        
        print(f"Line {line_num}: Unknown format, keys: {list(record.keys())}")
        return None
        
    except Exception as e:
        print(f"Line {line_num}: Error fixing format - {e}")
        return None

def validate_single_record(record: Dict[str, Any]) -> bool:
    """
    Validate a single record for Gemini fine-tuning format.
    
    Args:
        record (dict): Record to validate
        
    Returns:
        bool: True if valid
    """
    
    # Must have contents field
    if "contents" not in record:
        return False
    
    contents = record["contents"]
    
    # Contents must be a list
    if not isinstance(contents, list):
        return False
    
    # Must have at least one message
    if len(contents) == 0:
        return False
    
    # Validate each message
    for msg in contents:
        if not isinstance(msg, dict):
            return False
        
        # Must have role and parts
        if "role" not in msg or "parts" not in msg:
            return False
        
        # Role must be valid
        if msg["role"] not in ["user", "model"]:
            return False
        
        # Parts must be a list
        if not isinstance(msg["parts"], list):
            return False
        
        # Each part must have text
        for part in msg["parts"]:
            if not isinstance(part, dict) or "text" not in part:
                return False
    
    return True

def create_sample_gemini_format() -> Dict[str, Any]:
    """Create a sample record in correct Gemini format."""
    return {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "Generate a meal plan for a 30-year-old male with diabetes."}]
            },
            {
                "role": "model", 
                "parts": [{"text": "Here's a diabetes-friendly meal plan:\n\nBreakfast:\n- Oatmeal with berries\n- Greek yogurt\n\nLunch:\n- Grilled chicken salad\n- Whole grain roll\n\nDinner:\n- Baked salmon\n- Steamed vegetables\n- Brown rice"}]
            }
        ]
    }

def analyze_current_format(input_file: str) -> Dict[str, Any]:
    """
    Analyze the current format of the JSONL file.
    
    Args:
        input_file (str): Path to input file
        
    Returns:
        dict: Analysis results
    """
    
    analysis = {
        "total_lines": 0,
        "valid_json": 0,
        "has_contents": 0,
        "has_system_instruction": 0,
        "sample_keys": set(),
        "sample_record": None,
        "errors": []
    }
    
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                analysis["total_lines"] += 1
                line = line.strip()
                
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                    analysis["valid_json"] += 1
                    
                    if isinstance(record, dict):
                        analysis["sample_keys"].update(record.keys())
                        
                        if "contents" in record:
                            analysis["has_contents"] += 1
                        
                        if "systemInstruction" in record:
                            analysis["has_system_instruction"] += 1
                        
                        if analysis["sample_record"] is None:
                            analysis["sample_record"] = record
                    
                    # Only analyze first 10 records
                    if analysis["valid_json"] >= 10:
                        break
                        
                except json.JSONDecodeError as e:
                    analysis["errors"].append(f"Line {line_num}: {str(e)}")
                    if len(analysis["errors"]) >= 5:
                        break
    
    except Exception as e:
        analysis["errors"].append(f"File error: {str(e)}")
    
    return analysis

# Main execution
if __name__ == "__main__":
    input_file = "clinical_finetune_dataset_no_common_meds (1).jsonl"
    
    print("=== ANALYZING CURRENT FORMAT ===")
    analysis = analyze_current_format(input_file)
    
    print(f"Total lines: {analysis['total_lines']}")
    print(f"Valid JSON lines: {analysis['valid_json']}")
    print(f"Lines with 'contents' field: {analysis['has_contents']}")
    print(f"Lines with 'systemInstruction' field: {analysis['has_system_instruction']}")
    print(f"Sample keys found: {list(analysis['sample_keys'])}")
    
    if analysis['sample_record']:
        print(f"\nSample record structure:")
        print(json.dumps(analysis['sample_record'], indent=2)[:500] + "...")
    
    if analysis['errors']:
        print(f"\nErrors found:")
        for error in analysis['errors']:
            print(f"  - {error}")
    
    print(f"\n=== CREATING SAMPLE CORRECT FORMAT ===")
    sample = create_sample_gemini_format()
    print("Correct Gemini format should look like:")
    print(json.dumps(sample, indent=2))
    
    print(f"\n=== FIXING FILE FORMAT ===")
    success = validate_gemini_finetune_format(input_file)
    
    if success:
        print("✅ File successfully converted to Gemini format!")
    else:
        print("❌ Some records could not be converted. Check the output above for details.")
        print("\nTip: Manually review the invalid records and ensure they follow the correct format.")