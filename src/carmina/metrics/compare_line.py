import re
import json
from src.carmina.metrics.evaluator import calculate_metrics, get_all_metrics, calculate_average_metrics

def split_clinical_document(content):
    """
    Divide un documento clínico en líneas individuales basadas en los \n literales.
    
    Args:
        content (str): El contenido del documento clínico.
    
    Returns:
        list: Lista de líneas individuales.
    """
    return content.split('\n')

def _extract_tags_from_line(line, tag_pattern):
    """
    Extracts tags from a single line using the provided regex pattern.

    Args:
        line (str): The line to process.
        tag_pattern (re.Pattern): The compiled regex pattern for finding tags.

    Returns:
        list: A list of tags found in the line.
    """
    return tag_pattern.findall(line)

def extract_tags_from_files(content_a, content_b):
    """
    Reads two files, extracts tags in the format [**TAG**], and returns a dictionary 
    in the requested JSON format.

    Args:
        file_a_path (str): Path to the first file.
        file_b_path (str): Path to the second file.

    Returns:
        dict: A dictionary containing the extracted tags and their corresponding lines.
    """
    result = {}
    tag_pattern = re.compile(r'\[\*\*(.*?)\*\*\]')

    try:
        
        # Dividir contenido en líneas individuales
        lines_a = split_clinical_document(content_a)
        lines_b = split_clinical_document(content_b)
        
        # Procesar todas las líneas
        max_lines = max(len(lines_a), len(lines_b))
        for line_number in range(max_lines):
            line_a = lines_a[line_number] if line_number < len(lines_a) else ""
            line_b = lines_b[line_number] if line_number < len(lines_b) else ""
            
            tags_a = _extract_tags_from_line(line_a, tag_pattern)
            tags_b = _extract_tags_from_line(line_b, tag_pattern)

            metrics  = get_all_metrics(tags_a, tags_b)

            line_id = f"line_{line_number+1}"
            result[line_id] = {
                "textA": line_a,
                "textB": line_b,
                "labelA": tags_a,
                "labelB": tags_b,
                "metrics": metrics
            }

        # Reorganize the result to include average metrics
        average_metrics = calculate_metrics(content_a, content_b, result)
        final_result = {"metrics": average_metrics}
        final_result.update(result)
    except FileNotFoundError:
        print(f"Error: One or both files not found: {content_a}, {content_b}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    return final_result


def extract_all_metrics(ground_truth_identity_texts,
                        prediction_identity_texts, 
                        ground_truth_texts, 
                        prediction_texts, 
                        filenames, 
                        languages) -> dict:
    """
    Extracts all metrics from the two files and returns them in a dictionary.

    Args:
        content_a (str): Content of the first file.
        content_b (str): Content of the second file.

    Returns:
        dict: A dictionary containing all metrics.
    """
    data = {"metrics": {}}
    result = []
    for gt_identify, pred_identify, gt_text, pred_text, filename, language in zip(
            ground_truth_identity_texts, 
            prediction_identity_texts, 
            ground_truth_texts, 
            prediction_texts,
            filenames,
            languages): 
        
        # Call the function to extract tags
        extracted_data = {"id": filename, "language": language, "identify": {}, "label": {}}
        extracted_data["identify"]= extract_tags_from_files(gt_identify, pred_identify)
        extracted_data["label"] = extract_tags_from_files(gt_text, pred_text)
        result.append(extracted_data)
    data["files"] = result
    data["metrics"]["identify"] = calculate_average_metrics(result, "identify")
    data["metrics"]["label"] = calculate_average_metrics(result, "label")
    return data


# Example usage:
if __name__ == "__main__":
    file_a_path_example = 'a.txt'
    file_b_path_example = 'b.txt'

    extracted_data = extract_tags_from_files(file_a_path_example, file_b_path_example)