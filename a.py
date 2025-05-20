import re

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

def _append_remaining_lines_tags(
    file_handle, 
    tag_pattern, 
    results_dict, 
    starting_line_num, 
    is_file_a
):
    """
    Processes remaining lines from a file and adds their tags to the results dictionary.

    Args:
        file_handle (io.TextIOWrapper): The file handle to read lines from.
        tag_pattern (re.Pattern): The compiled regex pattern for finding tags.
        results_dict (dict): The dictionary to append tag results to.
        starting_line_num (int): The line number to start from for dictionary keys.
        is_file_a (bool): True if processing file_a, False for file_b.

    Returns:
        int: The next line number after processing all lines in this handle.
    """
    current_line_num = starting_line_num
    for line in file_handle:
        current_line_num += 1
        tags = _extract_tags_from_line(line, tag_pattern)
        line_key = f"line_{current_line_num}"
        if is_file_a:
            results_dict[line_key] = {"file_a_tags": tags, "file_b_tags": []}
        else:
            results_dict[line_key] = {"file_a_tags": [], "file_b_tags": tags}
    return current_line_num


def extract_tags_from_files(file_a_path, file_b_path):
    """
    Reads two files line by line, extracts tags in the format [**TAG**],
    and returns a dictionary detailing tags per line.

    Args:
        file_a_path (str): Path to the first file.
        file_b_path (str): Path to the second file.

    Returns:
        dict: A dictionary where keys are 'line_N' (e.g., 'line_1') and 
              values are dictionaries with 'file_a_tags' and 'file_b_tags' lists.
              Example:
              {
                "line_1": {"file_a_tags": ['TAG_A1'], "file_b_tags": ['TAG_B1']},
                "line_2": {"file_a_tags": [], "file_b_tags": ['TAG_B2']}
              }
              Returns None if an error occurs (e.g., file not found).
    """
    comparison_results = {}
    tag_pattern = re.compile(r'\[\*\*(.*?)\*\*\]')
    current_line_number = 0

    try:
        with open(file_a_path, 'r', encoding='utf-8') as file_a, \
             open(file_b_path, 'r', encoding='utf-8') as file_b:

            # Process lines present in both files
            for line_a_content, line_b_content in zip(file_a, file_b):
                current_line_number += 1
                tags_a = _extract_tags_from_line(line_a_content, tag_pattern)
                tags_b = _extract_tags_from_line(line_b_content, tag_pattern)
                comparison_results[f"line_{current_line_number}"] = {
                    "file_a_tags": tags_a,
                    "file_b_tags": tags_b
                }
            
            # Process remaining lines from file_a, if any
            current_line_number = _append_remaining_lines_tags(
                file_a, tag_pattern, comparison_results, current_line_number, is_file_a=True
            )
            
            # Process remaining lines from file_b, if any
            _append_remaining_lines_tags(
                file_b, tag_pattern, comparison_results, current_line_number, is_file_a=False
            )

    except FileNotFoundError:
        print(f"Error: One or both files not found: {file_a_path}, {file_b_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    return comparison_results

# Example usage:
if __name__ == "__main__":
    # Create dummy files for demonstration if they don't exist
    file_a_path_example = 'a.txt'
    file_b_path_example = 'b.txt'


    extracted_data = extract_tags_from_files(file_a_path_example, file_b_path_example)

    if extracted_data:
        for line_key, tags_info in extracted_data.items():
            print(f"{line_key}:")
            print(f"  Etiquetas de '{file_a_path_example}': {tags_info['file_a_tags']}")
            print(f"  Etiquetas de '{file_b_path_example}': {tags_info['file_b_tags']}")
    else:
        print("No se pudieron extraer datos o los archivos no se encontraron.")
