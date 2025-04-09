import re
from typing import List, Dict, Any, Tuple

def extract_labels(identify_text: str) -> List[Dict[str, Any]]:
    """
    Extracts PHI labels from identify text using the [**ORIGINAL_VALUE**] pattern.
    
    Args:
        identify_text (str): Text with identified entities.
        
    Returns:
        List[Dict[str, Any]]: List of extracted labels.
    """
    labels = []
    pattern = r'\[\*\*(.*?)\*\*\]'
    
    for match in re.finditer(pattern, identify_text):
        entity_value = match.group(1)
        
        labels.append({
            'type': 'PHI',  # Generic type as we don't have specific types in identify format
            'text': entity_value,
            'start': match.start(),
            'end': match.end()
        })
        
    return labels

def extract_clean_text_and_labels(identify_text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Extracts clean text and labels from identify text.
    
    Args:
        identify_text (str): Text with [**value**] annotations
        
    Returns:
        Tuple[str, List[Dict[str, Any]]]: Clean text and labels
    """
    clean_text = identify_text
    labels = []
    pattern = r'\[\*\*(.*?)\*\*\]'
    
    # Find all annotations
    offset = 0
    for match in re.finditer(pattern, identify_text):
        entity_text = match.group(1)
        start_pos = match.start() - offset
        
        # Replace the annotation with the original text
        clean_text = clean_text.replace(match.group(0), entity_text, 1)
        
        # Update the offset for subsequent matches
        offset += len(match.group(0)) - len(entity_text)
        
        # Calculate end position in clean text
        end_pos = start_pos + len(entity_text)
        
        # Add to labels
        labels.append({
            'type': 'PHI',  # Generic type
            'text': entity_text,
            'start': start_pos,
            'end': end_pos
        })
        
    return clean_text, labels