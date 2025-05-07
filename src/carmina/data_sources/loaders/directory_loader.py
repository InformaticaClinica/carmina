import os
from typing import List, Dict, Any
from ..exceptions import DataLoadError
from ..extractors import (
    extract_labels_from_masked_text,
    extract_value_from_masked_text)

from .file_loaders import load_json_file, load_csv_file, load_txt_file

def load_directory(dir_path: str) -> List[Dict[str, Any]]:
    """
    Loads all supported files from a directory, combining them into a single dataset.
    
    Supports three types of directory structures:
    1. Raw text + masked text (entity types)
    2. Raw text + identify text (marked entities)
    3. Standard directory with individual files
    
    Args:
        dir_path (str): Path to the directory containing data files.
        
    Returns:
        List[Dict[str, Any]]: Combined records from all files.
    """
    records = []
    
    txt_raw_path = os.path.join(dir_path, 'txt', 'raw')
    txt_masked_path = os.path.join(dir_path, 'txt', 'masked')
    txt_identify_path = os.path.join(dir_path, 'txt', 'identify')
    
    # Case: All three formats available (raw + masked + identify)
    if os.path.exists(txt_raw_path) and os.path.exists(txt_masked_path) and os.path.exists(txt_identify_path):
        records = _load_all_formats(txt_raw_path, txt_masked_path, txt_identify_path)
    # Case 1: Raw + Masked structure
    elif os.path.exists(txt_raw_path) and os.path.exists(txt_masked_path):
        records = _load_raw_masked_structure(txt_raw_path, txt_masked_path)
    # Case 2: Raw + Identify structure
    elif os.path.exists(txt_raw_path) and os.path.exists(txt_identify_path):
        records = _load_raw_identify_structure(txt_raw_path, txt_identify_path)
    # Case 3: Only identify structure exists
    elif os.path.exists(txt_identify_path):
        records = _load_identify_structure(txt_identify_path)
    # Case 4: Standard directory with files
    else:
        records = _load_standard_directory(dir_path)
    
    return records

def _load_all_formats(txt_raw_path: str, txt_masked_path: str, txt_identify_path: str) -> List[Dict[str, Any]]:
    """
    Loads data from raw, masked and identify text directories.
    
    Args:
        txt_raw_path (str): Path to directory with raw texts.
        txt_masked_path (str): Path to directory with masked texts.
        txt_identify_path (str): Path to directory with identified texts.
        
    Returns:
        List[Dict[str, Any]]: List of records in the required format.
    """
    records = []
    
    for filename in sorted(os.listdir(txt_raw_path)):
        if not filename.endswith('.txt'):
            continue
            
        raw_path = os.path.join(txt_raw_path, filename)
        masked_path = os.path.join(txt_masked_path, filename)
        identify_path = os.path.join(txt_identify_path, filename)
        
        with open(raw_path, 'r', encoding='utf-8') as f_raw:
            original_text = f_raw.read()
            
        with open(masked_path, 'r', encoding='utf-8') as f_masked:
            masked_text = f_masked.read()
            
        with open(identify_path, 'r', encoding='utf-8') as f_identify:
            identify_text = f_identify.read()
            
        # Extract labels from masked text but use original text values
        labels_with_types = extract_labels_from_masked_text(masked_text)
        # Extract labels from masked text but use original text values
        labels_with_values = extract_labels_from_masked_text(identify_text)
        
        # Map the extracted labels to get the original text values
        labels = []
        for label_types, label_value in zip(labels_with_types, labels_with_values):
            # Use the original text value from the identify text
            entity_type = label_types["text"]
            text_value = label_value["text"]
            entity_start = label_types["start"]
            entity_end = label_types["end"]
            
            labels.append({
                "type": entity_type,
                "text": text_value,
                "start": entity_start,
                "end": entity_end
            })
            
        records.append({
            'id': filename,
            'text': original_text,
            'identify': identify_text,  # Using the spelling from user's request
            'masked_text': masked_text,  # Using the spelling from user's request
            'labels': labels
        })
    
    return records

def _load_raw_masked_structure(txt_raw_path: str, txt_masked_path: str) -> List[Dict[str, Any]]:
    """
    Loads data from raw and masked text directories.
    
    Args:
        txt_raw_path (str): Path to directory with raw texts.
        txt_masked_path (str): Path to directory with masked texts.
        
    Returns:
        List[Dict[str, Any]]: List of records in the required format.
    """
    records = []
    
    for filename in sorted(os.listdir(txt_raw_path)):
        if not filename.endswith('.txt'):
            continue
            
        raw_path = os.path.join(txt_raw_path, filename)
        masked_path = os.path.join(txt_masked_path, filename)
        
        # Skip if masked file doesn't exist
        if not os.path.exists(masked_path):
            continue
            
        with open(raw_path, 'r', encoding='utf-8') as f_raw:
            original_text = f_raw.read()
            
        with open(masked_path, 'r', encoding='utf-8') as f_masked:
            masked_text = f_masked.read()
            
        # Extract labels from masked text
        labels = extract_labels_from_masked_text(masked_text)
        value = extract_value_from_masked_text(masked_text, original_text)
        print(value)
        
        records.append({
            'id': filename,
            'text': original_text,
            'masked_text': masked_text,
            'labels': labels
        })
    
    return records

# def _find_entity_value(original_text: str, entity_type: str, masked_text: str, start: int, end: int) -> str:
    # """
    # Attempts to find the original entity value from the masked text.
    # This is a heuristic approach that tries to align positions.
    
    # Args:
        # original_text: The original text with real values
        # entity_type: The entity type (e.g., "PROFESION")
        # masked_text: The masked text with [**TYPE**] patterns
        # start: Start position in masked text
        # end: End position in masked text
        
    # Returns:
        # str: The corresponding value from the original text
    # """
    # # This is a simplified implementation - in practice, you'd need
    # # more sophisticated alignment between masked and original text
    
    # # Try to find context before the entity
    # context_start = max(0, start - 20)
    # context_before = masked_text[context_start:start]
    
    # # Remove the entity tag itself
    # tag = f"[**{entity_type}**]"
    
    # # Find the context in the original text
    # if context_before in original_text:
        # pos = original_text.find(context_before) + len(context_before)
        
        # # Extract a reasonable span of text after the context
        # span_end = pos + 40
        # potential_value = original_text[pos:span_end].strip()
        
        # # Try to find a reasonable stopping point for the entity
        # for separator in [' ', '\n', '.', ',', ';', ':', ')', ']']:
            # if separator in potential_value:
                # potential_value = potential_value.split(separator)[0].strip()
                # break
                
        # return potential_value
    
    # # Fallback - just return the entity type as the value
    # return entity_type

# Keep the rest of the functions with appropriate modifications to return the desired format...