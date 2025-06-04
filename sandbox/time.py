#!/usr/bin/env python3
"""
Script para mover exactamente un año las fechas que se encuentran 
en los archivos de la carpeta rd_med_content_anonymized.

Las fechas están en formato [**DD.MM.YYYY**] o [**DD/MM/YYYY**]
"""

import os
import re
from datetime import datetime, timedelta
from pathlib import Path


def add_one_year_to_date(date_str, separator):
    """
    Suma un año a una fecha dada.
    
    Args:
        date_str: Fecha en formato DD.MM.YYYY o DD/MM/YYYY
        separator: '.' o '/'
    
    Returns:
        Nueva fecha con un año más en el mismo formato
    """
    try:
        # Parsear la fecha
        day, month, year = map(int, date_str.split(separator))
        
        # Crear objeto datetime
        original_date = datetime(year, month, day)
        
        # Sumar un año
        # Manejar años bisiestos (29 de febrero)
        try:
            new_date = original_date.replace(year=year + 1)
        except ValueError:
            # Si es 29 de febrero en año bisiesto, mover a 28 de febrero
            new_date = original_date.replace(year=year + 1, day=28)
        
        # Formatear de vuelta
        return f"{new_date.day:02d}{separator}{new_date.month:02d}{separator}{new_date.year}"
    
    except (ValueError, TypeError) as e:
        print(f"Error procesando fecha {date_str}: {e}")
        return date_str  # Devolver original si hay error


def process_file(file_path):
    """
    Procesa un archivo, reemplazando todas las fechas encontradas
    sumándoles un año.
    """
    print(f"Procesando: {file_path}")
    
    try:
        # Leer el contenido del archivo
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Patrones para fechas con punto y con barra
        pattern_dot = r'\[\*\*(\d{1,2})\.(\d{1,2})\.(\d{4})\*\*\]'
        pattern_slash = r'\[\*\*(\d{1,2})/(\d{1,2})/(\d{4})\*\*\]'
        
        # Contador de reemplazos
        replacements = 0
        
        # Función para reemplazar fechas con punto
        def replace_dot_date(match):
            nonlocal replacements
            day, month, year = match.groups()
            original_date = f"{day}.{month}.{year}"
            new_date = add_one_year_to_date(original_date, '.')
            replacements += 1
            return f"[**{new_date}**]"
        
        # Función para reemplazar fechas con barra
        def replace_slash_date(match):
            nonlocal replacements
            day, month, year = match.groups()
            original_date = f"{day}/{month}/{year}"
            new_date = add_one_year_to_date(original_date, '/')
            replacements += 1
            return f"[**{new_date}**]"
        
        # Aplicar reemplazos
        content = re.sub(pattern_dot, replace_dot_date, content)
        content = re.sub(pattern_slash, replace_slash_date, content)
        
        # Escribir el archivo modificado
        if replacements > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✓ {replacements} fechas actualizadas")
        else:
            print(f"  - No se encontraron fechas para actualizar")
    
    except Exception as e:
        print(f"  ✗ Error procesando {file_path}: {e}")


def main():
    """
    Función principal que procesa todos los archivos en la carpeta
    rd_med_content_anonymized
    """
    # Ruta de la carpeta
    folder_path = Path("/Users/petterpenafiel/Documents/sandbox/rd_med_nhc_only_anonymized")
    
    if not folder_path.exists():
        print(f"Error: La carpeta {folder_path} no existe")
        return
    
    # Obtener todos los archivos .txt en la carpeta
    txt_files = list(folder_path.glob("*.txt"))
    
    if not txt_files:
        print(f"No se encontraron archivos .txt en {folder_path}")
        return
    
    print(f"Encontrados {len(txt_files)} archivos para procesar")
    print("=" * 50)
    
    # Procesar cada archivo
    total_files = len(txt_files)
    for i, file_path in enumerate(txt_files, 1):
        print(f"[{i}/{total_files}]", end=" ")
        process_file(file_path)
    
    print("=" * 50)
    print("✓ Procesamiento completado")


if __name__ == "__main__":
    main()
