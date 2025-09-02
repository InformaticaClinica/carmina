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

def clean_lines_array(lines):
    """
    Limpia un array de líneas eliminando líneas vacías o que solo contengan espacios/períodos.
    
    Args:
        lines (list): Lista de líneas a limpiar.
    
    Returns:
        list: Lista de líneas limpias sin elementos vacíos.
    """
    cleaned_lines = []
    for line in lines:
        line_clean = line.strip()
        # Solo mantener líneas que tengan contenido válido (no vacías ni solo períodos)
        if line_clean and line_clean != ".":
            cleaned_lines.append(line)
    return cleaned_lines

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

def find_best_line_alignment(lines_a, lines_b, tag_pattern):
    """
    Encuentra el mejor alineamiento entre dos arrays de líneas usando programación dinámica.
    Detecta líneas faltantes o insertadas basándose en similitud semántica del contenido.
    """
    def _normalize_content(line):
        """Normaliza el contenido de una línea para comparación semántica."""
        # Remover tags y números de lista, limpiar espacios
        clean = re.sub(r'\[\*\*.*?\*\*\]', '', line).strip().lower()
        clean = re.sub(r'^\d+\.\s*', '', clean).strip()
        return clean
    
    def _calculate_similarity(line1, line2):
        """Calcula similitud semántica entre dos líneas normalizadas."""
        clean1, clean2 = _normalize_content(line1), _normalize_content(line2)
        
        if not clean1 or not clean2:
            return 0.0
            
        words1, words2 = set(clean1.split()), set(clean2.split())
        
        # Manejo especial para líneas muy cortas
        if not words1 or not words2:
            return 0.8 if clean1 == clean2 else 0.0
        
        # Cálculo de similitud basado en intersección/unión de palabras (Jaccard)
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        jaccard_sim = intersection / union if union > 0 else 0.0
        
        # Factor de ajuste por longitud similar
        len_factor = min(len(clean1), len(clean2)) / max(len(clean1), len(clean2))
        
        # Alta confianza para contenido muy similar
        if jaccard_sim > 0.9:
            return 0.95
        
        return jaccard_sim * (0.8 + 0.2 * len_factor)
    
    # Configuración de la matriz de programación dinámica
    m, n = len(lines_a), len(lines_b)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    trace = [[None] * (n + 1) for _ in range(m + 1)]
    
    # Construcción de la matriz usando programación dinámica
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            similarity = _calculate_similarity(lines_a[i-1], lines_b[j-1])
            
            # Evaluar todas las opciones posibles
            options = [
                (dp[i-1][j-1] + similarity, 'match'),  # Alinear ambas líneas
                (dp[i-1][j], 'skip_a'),                # Saltar línea en A
                (dp[i][j-1], 'skip_b')                 # Saltar línea en B
            ]
            
            best_score, best_action = max(options, key=lambda x: x[0])
            dp[i][j] = best_score
            trace[i][j] = best_action
    
    # Reconstrucción del alineamiento óptimo
    alignments = []
    i, j = m, n
    
    while i > 0 or j > 0:
        action = trace[i][j] if i > 0 and j > 0 else None
        
        if action == 'match':
            similarity = _calculate_similarity(lines_a[i-1], lines_b[j-1])
            if similarity > 0.4:  # Umbral mínimo para match válido
                alignments.append((i-1, j-1))
            else:
                # Si la similitud es muy baja, tratar como líneas separadas
                alignments.extend([(i-1, -1), (-1, j-1)])
            i, j = i-1, j-1
        elif action == 'skip_a' or i > 0:
            alignments.append((i-1, -1))
            i -= 1
        else:  # skip_b or j > 0
            alignments.append((-1, j-1))
            j -= 1
    
    return list(reversed(alignments))

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
        
        # Limpiar arrays eliminando líneas vacías ANTES del procesamiento
        lines_a_clean = clean_lines_array(lines_a)
        lines_b_clean = clean_lines_array(lines_b)
        
        # Detectar si los arrays tienen diferencias significativas
        size_diff = abs(len(lines_a_clean) - len(lines_b_clean))
        needs_alignment = size_diff > 0 or len(lines_a_clean) != len(lines_b_clean)
        
        def _process_line_pair(line_a, line_b, line_id):
            """Procesa un par de líneas y genera la entrada de resultado."""
            if not line_a.strip() or not line_b.strip():
                return None  # Skip empty lines
                
            tags_a = _extract_tags_from_line(line_a, tag_pattern)
            tags_b = _extract_tags_from_line(line_b, tag_pattern)
            metrics = get_all_metrics(tags_a, tags_b)
            
            return {
                "textA": line_a,
                "textB": line_b,
                "labelA": tags_a,
                "labelB": tags_b,
                "metrics": metrics
            }
        
        if needs_alignment:
            # Usar alineamiento inteligente para documentos con diferencias estructurales
            alignments = find_best_line_alignment(lines_a_clean, lines_b_clean, tag_pattern)
            
            line_counter = 1
            for i_a, i_b in alignments:
                line_a = lines_a_clean[i_a] if i_a >= 0 else ""
                line_b = lines_b_clean[i_b] if i_b >= 0 else ""
                
                processed = _process_line_pair(line_a, line_b, f"line_{line_counter}")
                if processed:
                    result[f"line_{line_counter}"] = processed
                    line_counter += 1
        else:
            # Procesamiento secuencial para documentos con estructura similar
            max_lines = max(len(lines_a_clean), len(lines_b_clean))
            
            for line_number in range(max_lines):
                line_a = lines_a_clean[line_number] if line_number < len(lines_a_clean) else ""
                line_b = lines_b_clean[line_number] if line_number < len(lines_b_clean) else ""
                
                processed = _process_line_pair(line_a, line_b, f"line_{line_number+1}")
                if processed:
                    result[f"line_{line_number+1}"] = processed

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