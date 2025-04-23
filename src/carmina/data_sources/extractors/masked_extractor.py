import re
from typing import List, Dict, Any

def extract_labels(masked_text: str) -> List[Dict[str, Any]]:
    """
    Extracts PHI labels from masked text using the [**ENTITY_TYPE**] pattern.
    
    Args:
        masked_text (str): Text with masked entities.
        
    Returns:
        List[Dict[str, Any]]: List of extracted labels.
    """
    labels = []
    pattern = r'\[\*\*(.*?)\*\*\]'
    
    for match in re.finditer(pattern, masked_text):
        entity = match.group(1)
        # Format like [**PHI_TYPE**]
        value = entity.strip()
        
        # Calculate the start and end positions in the original text
        start_pos = len(re.sub(r'[\n\t\r]', '', masked_text[:match.start()]))
        end_pos = start_pos + len(re.sub(r'[\n\t\r]', '', match.group(0)))
        
        labels.append({
            'text': value,
            'start': start_pos,
            'end': end_pos
        })
        
    return labels