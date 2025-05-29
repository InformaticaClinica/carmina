import os
import json
import pandas as pd
from typing import List, Dict, Any
from ..exceptions import DataLoadError

def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Loads records from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file.
        
    Returns:
        List[Dict[str, Any]]: List of records.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Handle both list and dictionary formats
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'records' in data:
            return data['records']
        elif isinstance(data, dict):
            # Single record
            return [data]
        else:
            raise DataLoadError(f"Unexpected JSON structure in {file_path}")
    except json.JSONDecodeError:
        raise DataLoadError(f"Invalid JSON format in {file_path}")


def load_csv_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Loads records from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        List[Dict[str, Any]]: List of records.
    """
    try:
        df = pd.read_csv(file_path)
        records = df.to_dict('records')
        
        # Ensure each record has the required fields
        for record in records:
            if 'text' not in record and 'content' in record:
                record['text'] = record['content']
            if 'labels' not in record:
                record['labels'] = []
                
        return records
    except Exception as e:
        raise DataLoadError(f"Failed to load CSV file {file_path}: {str(e)}")



def load_txt_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Loads a single record from a text file.
    
    Args:
        file_path (str): Path to the text file.
        
    Returns:
        List[Dict[str, Any]]: List containing a single record.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        filename = os.path.basename(file_path)
        return [{
            'id': filename,
            'text': content,
            'labels': []  # Default empty labels for plain text
        }]
    except Exception as e:
        raise DataLoadError(f"Failed to load text file {file_path}: {str(e)}")