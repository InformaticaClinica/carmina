import re

def extract_labels(texto_anonimizado, texto_original, tamano_contexto=30):
    """
    Encuentra las sustituciones entre un texto anonimizado y el texto original.
    
    Args:
        texto_anonimizado: Texto con etiquetas en formato [ETIQUETA]
        texto_original: Texto original sin anonimizar
        tamano_contexto: Cantidad de caracteres de contexto a considerar (default: 30)
        
    Returns:
        Diccionario con las etiquetas y sus valores originales
    """
    # Buscar todas las etiquetas en formato [**ETIQUETA**]
    etiquetas = re.findall(r'\[\*\*([A-Z_]+)\*\*\]', texto_anonimizado)
    
    # Diccionario para almacenar los resultados
    resultado = {}
    
    # Para cada etiqueta encontrada
    for etiqueta in etiquetas:
        # Buscar todas las ocurrencias de esta etiqueta
        for match in re.finditer(r'\[' + etiqueta + r'\]', texto_anonimizado):
            # Obtener posición de la etiqueta
            pos = match.start()
            etiqueta_completa = match.group(0)  # [ETIQUETA] completa
            
            # Extraer contexto alrededor de la etiqueta
            contexto_antes = texto_anonimizado[max(0, pos-tamano_contexto):pos]
            contexto_despues = texto_anonimizado[pos+len(etiqueta_completa):min(len(texto_anonimizado), pos+len(etiqueta_completa)+tamano_contexto)]
            
            # Escapar caracteres especiales en los contextos
            contexto_antes_escaped = re.escape(contexto_antes)
            contexto_despues_escaped = re.escape(contexto_despues)
            
            try:
                # Construir patrón para buscar en el texto original
                patron = f"{contexto_antes_escaped}(.*?){contexto_despues_escaped}"
                
                # Buscar en el texto original
                coincidencia = re.search(patron, texto_original)
                if coincidencia:
                    valor = coincidencia.group(1).strip()
                    
                    # Guardar el resultado
                    if etiqueta not in resultado:
                        resultado[etiqueta] = valor
                    # Si ya existe pero es diferente, convertir a lista
                    elif isinstance(resultado[etiqueta], str) and resultado[etiqueta] != valor:
                        resultado[etiqueta] = [resultado[etiqueta], valor]
                    # Si ya es una lista y el valor no está en ella, agregarlo
                    elif isinstance(resultado[etiqueta], list) and valor not in resultado[etiqueta]:
                        resultado[etiqueta].append(valor)
            except re.error as e:
                print(f"Error en la expresión regular para la etiqueta {etiqueta}: {e}")
    
    return resultado