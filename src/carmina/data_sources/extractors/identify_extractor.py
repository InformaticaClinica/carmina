import re

def extract_labels(anonymized_text, original_text, context_size=30):
    """
    Finds substitutions between an anonymized text and the original text.
    
    Args:
        anonymized_text: Text with labels in the format [LABEL]
        original_text: Original text without anonymization
        context_size: Number of context characters to consider (default: 30)
        
    Returns:
        Dictionary with the labels and their original values
    """
    # Find all labels in the format [**LABEL**]
    labels = re.findall(r'\[\*\*([A-Z_]+)\*\*\]', anonymized_text)
    
    # Dictionary to store the results
    result = {}
    
    # For each label found
    for label in labels:
        # Find all occurrences of this label
        for match in re.finditer(r'\[' + label + r'\]', anonymized_text):
            # Get the position of the label
            pos = match.start()
            full_label = match.group(0)  # Complete [LABEL]
            
            # Extract context around the label
            context_before = anonymized_text[max(0, pos-context_size):pos]
            context_after = anonymized_text[pos+len(full_label):min(len(anonymized_text), pos+len(full_label)+context_size)]
            
            # Escape special characters in the contexts
            context_before_escaped = re.escape(context_before)
            context_after_escaped = re.escape(context_after)
            
            try:
                # Build pattern to search in the original text
                pattern = f"{context_before_escaped}(.*?){context_after_escaped}"
                
                # Search in the original text
                match_original = re.search(pattern, original_text)
                if match_original:
                    value = match_original.group(1).strip()
                    
                    # Save the result
                    if label not in result:
                        result[label] = value
                    # If it already exists but is different, convert to list
                    elif isinstance(result[label], str) and result[label] != value:
                        result[label] = [result[label], value]
                    # If it's already a list and the value is not in it, add it
                    elif isinstance(result[label], list) and value not in result[label]:
                        result[label].append(value)
            except re.error as e:
                print(f"Error in the regular expression for label {label}: {e}")
    
    return result