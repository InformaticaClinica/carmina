import os
import pandas as pd
from typing import List, Dict, Any
from ..exceptions import DataLoadError
from ..extractors import (
    extract_labels_from_masked_text,
    extract_value_from_masked_text)

from .file_loaders import load_json_file, load_csv_file, load_txt_file
import re

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
    
    txt_raw_path = os.path.join(dir_path, 'txt', 'raw/')
    txt_masked_path = os.path.join(dir_path, 'txt', 'masked/')
    text_ann_path = os.path.join(dir_path, 'txt', 'ann/')
    txt_identify_path = os.path.join(dir_path, 'txt', 'identify/')
    
    if os.getenv('ANONYMIZATION_MODE') == 'substitute':
            return _load_directory(dir_path)

    if not os.path.exists(txt_identify_path):
        _create_identify_format(txt_raw_path, text_ann_path, txt_identify_path)

    if os.path.exists(txt_raw_path) and os.path.exists(txt_masked_path) and os.path.exists(txt_identify_path):
        records = _load_all_formats(txt_raw_path, txt_masked_path, txt_identify_path)
    else:
        raise DataLoadError(f"Unsupported directory structure: {dir_path}")    
    return records

def _create_identify_format(txt_raw_path, text_ann_path, txt_identify_path):
    """
    Creates the identify format from raw and masked formats.
    
    Args:
        txt_raw_path (str): Path to directory with raw texts.
        txt_masked_path (str): Path to directory with masked texts.
        text_ann_path (str): Path to directory with annymized texts.
        
    Returns:
        None
    """
    #Create identify directory if it doesn't exist
    if not os.path.exists(txt_identify_path):
        os.makedirs(txt_identify_path)
    
    for filename_ann in sorted(os.listdir(text_ann_path)):
        base_name = os.path.splitext(filename_ann)[0]
        # Leer el archivo línea por línea
        ann_df = []
        with open(text_ann_path + filename_ann, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    id_, label_info, text = parts
                    label_parts = label_info.split(' ')
                    label = label_parts[0]
                    init = label_parts[1]
                    end = label_parts[2]
                    ann_df.append([id_, label, init, end, text])
        ann_df = pd.DataFrame(ann_df, columns=['id','label', 'init', 'end', 'text'])
        # Cambia el tipo de la columna 'init' a int
        ann_df['init'] = ann_df['init'].astype(int)
        ann_df = ann_df.sort_values(by='init')

        # Lee el texto original
        with open(txt_raw_path + base_name + '.txt', encoding='utf-8') as f:
            raw_text = f.read()

        # Reemplaza cada palabra por el formato [**WORD**] solo si no está ya enmascarada
        for word in ann_df['text']:
            if pd.notnull(word):
                # Escapa la palabra para expresiones regulares
                escaped_word = re.escape(str(word))
                # Solo reemplaza si no está ya entre [** **]
                pattern = rf'(?<!\[\*\*){escaped_word}(?!\*\*\])'
                raw_text = re.sub(pattern, f"[**{word}**]", raw_text)

        # Ahora raw_text tiene las palabras reemplazadas
        # Write the text with identified words to the identify path
        with open(os.path.join(txt_identify_path, f"{base_name}.txt"), 'w', encoding='utf-8') as f:
            f.write(raw_text)
        
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

def _load_directory(directory_path: str) -> List[Dict[str, Any]]:
    """
    Loads all records from a directory containing text files.
    
    Args:
        directory_path (str): Path to the directory.
        
    Returns:
        List[Dict[str, Any]]: List of records loaded from text files in the directory.
    """
    records = []
    
    for filename in sorted(os.listdir(directory_path)):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            records.extend(load_txt_file(file_path))
    return records