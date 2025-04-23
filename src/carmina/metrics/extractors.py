import re
import spacy
from typing import List, Dict, Any, Tuple, Optional

# Lazy-load spaCy model
_nlp = None

def get_nlp_model():
    """Lazy-load the spaCy model when needed."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("es_core_news_md")
    return _nlp

def extract_labels_from_masked_text(masked_text: str) -> List[str]:
    """
    Extract PHI labels from masked text using the [**ENTITY**] pattern.
    
    Args:
        masked_text: Text with entities marked as [**ENTITY**]
        
    Returns:
        List[str]: List of extracted entity labels
    """
    return re.findall(r'\[\*\*(.*?)\*\*\]', masked_text)

def remove_adverbs_determinants(text: str) -> str:
    """
    Remove adverbs and determinants from text using spaCy.
    
    Args:
        text: Input text
        
    Returns:
        str: Text with adverbs and determinants removed
    """
    nlp = get_nlp_model()
    doc = nlp(text)
    tokens_filtered = [token.text for token in doc if token.pos_ not in ('ADP', 'DET')]
    return ' '.join(tokens_filtered)

def detect_language(text: str) -> str:
    """
    Detect the language of a text.
    
    Currently using a mapping file lookup. Could be enhanced with language detection.
    
    Args:
        filename: Filename to lookup in the mapping file
        
    Returns:
        str: Language code
    """
    import pandas as pd
    
    # This is a placeholder - in real implementation, you'd want a more robust approach
    # that doesn't depend on filenames but uses actual language detection
    try:
        mapping_df = pd.read_csv('./data/carmen/CARMEN1_mappings.tsv', sep='\t', index_col='filename')
        filename = text.split('.')[0]  # Very naive - this should be improved
        return mapping_df.loc[filename, 'language']
    except:
        # Fallback to actual language detection
        # This would require a proper language detection library
        return "es"  # Default to Spanish