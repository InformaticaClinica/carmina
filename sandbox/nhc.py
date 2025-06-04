import hashlib
import os
import re
import shutil

def is_date_format(text):
    date_patterns = [
        r'^\d{1,2}\.\d{1,2}\.\d{4}$',
        r'^\d{1,2}/\d{1,2}/\d{4}$',
        r'^\d{1,2}-\d{1,2}-\d{4}$',
        r'^\d{4}-\d{1,2}-\d{1,2}$',
        r'^\d{1,2}\.\d{1,2}\.\d{2}$',
    ]
    
    for pattern in date_patterns:
        if re.match(pattern, text):
            return True
    return False

def extract_nhc_numbers(content):
    # Buscar ambos formatos: NHC: [**número**] y [**NHC: número**]
    pattern1 = r'NHC:\s*\[\*\*(\d+)\*\*\]'  # NHC: [**número**]
    pattern2 = r'\[\*\*NHC:\s*(\d+)\*\*\]'    # [**NHC: número**]
    
    matches1 = re.findall(pattern1, content)
    matches2 = re.findall(pattern2, content)
    
    all_matches = set(matches1 + matches2)
    return all_matches

def create_nhc_mapping(all_nhc_numbers):
    mapping = {}
    for nhc in all_nhc_numbers:
        hasher = hashlib.sha256()
        hasher.update(nhc.encode('utf-8'))
        hashed = hasher.hexdigest()[:12]
        mapping[nhc] = hashed
    return mapping

def anonymize_nhc_only(content, nhc_mapping):
    # Reemplazar ambos formatos de NHC
    
    # Formato 1: NHC: [**número**] -> NHC: [**hash**]
    def replace_nhc_format1(match):
        nhc_number = match.group(1)
        if nhc_number in nhc_mapping:
            return f"NHC: [**{nhc_mapping[nhc_number]}**]"
        return match.group(0)
    
    # Formato 2: [**NHC: número**] -> [**NHC: hash**]
    def replace_nhc_format2(match):
        nhc_number = match.group(1)
        if nhc_number in nhc_mapping:
            return f"[**NHC: {nhc_mapping[nhc_number]}**]"
        return match.group(0)
    
    # Aplicar ambos reemplazos
    pattern1 = r'NHC:\s*\[\*\*(\d+)\*\*\]'  # NHC: [**número**]
    pattern2 = r'\[\*\*NHC:\s*(\d+)\*\*\]'    # [**NHC: número**]
    
    content = re.sub(pattern1, replace_nhc_format1, content)
    content = re.sub(pattern2, replace_nhc_format2, content)
    
    return content

def main():
    source_folder = "rd_med"
    # No necesitamos carpeta de destino, sobreescribiremos los originales
    
    print("Extrayendo números NHC...")
    all_nhc = set()
    files = [f for f in os.listdir(source_folder) if f.endswith('.txt')]
    
    for filename in files:
        with open(os.path.join(source_folder, filename), 'r') as f:
            content = f.read()
            nhc_numbers = extract_nhc_numbers(content)
            all_nhc.update(nhc_numbers)
    
    print(f"Encontrados {len(all_nhc)} números NHC únicos: {sorted(all_nhc)}")
    
    nhc_mapping = create_nhc_mapping(all_nhc)
    
    # Guardar mapeo
    with open("nhc_mapping.txt", 'w') as f:
        f.write("NHC_Original\tNHC_Anonimizado\n")
        f.write("-" * 40 + "\n")
        for original, hashed in nhc_mapping.items():
            f.write(f"{original}\t{hashed}\n")
    
    print("Procesando y sobreescribiendo documentos originales...")
    for filename in files:
        file_path = os.path.join(source_folder, filename)
        
        # Leer contenido original
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Anonimizar
        anonymized = anonymize_nhc_only(content, nhc_mapping)
        
        # Sobreescribir el archivo original
        with open(file_path, 'w') as f:
            f.write(anonymized)
        
        print(f"Sobreescrito: {filename}")
    
    print(f"\nCompletado! Documentos originales sobreescritos con NHCs anonimizados.")
    print(f"Fechas y departamentos preservados.")
    print(f"Mapeo en: nhc_mapping.txt")
    
    return nhc_mapping

if __name__ == "__main__":
    main()
