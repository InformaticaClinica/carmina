import os
from typing import List, Dict, Any

from data_sources.exceptions import DataLoadError
from .loaders import load_directory, load_json_file, load_csv_file, load_txt_file

def load_dataset(input_path: str) -> List[Dict[str, Any]]:
    """
    Loads data from the specified input path and returns it as a list of record dictionaries.
    
    The function handles different file types (txt, json, csv) and directory structures,
    converting them into a unified format suitable for the anonymization pipeline.
    
    Args:
        input_path (str): Path to the input data. Can be a single file or directory.
        
    Returns:
        List[Dict[str, Any]]: A list of records, where each record is a dictionary 
        containing at minimum 'text' and 'labels' fields.
        
    Raises:
        DataLoadError: If the data cannot be loaded or has an unsupported format.
    """
    if not os.path.exists(input_path):
        raise DataLoadError(f"Input path does not exist: {input_path}")
    
    # Handle directory input
    if os.path.isdir(input_path):
        return load_directory(input_path)
    
    # Handle file input
    file_ext = os.path.splitext(input_path)[1].lower()
    if file_ext == '.json':
        return load_json_file(input_path)
    elif file_ext == '.csv':
        return load_csv_file(input_path)
    elif file_ext == '.txt':
        return load_txt_file(input_path)
    else:
        raise DataLoadError(f"Unsupported file format: {file_ext}")