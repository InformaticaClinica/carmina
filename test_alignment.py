#!/usr/bin/env python3

from src.carmina.metrics.compare_line import extract_tags_from_files

# Test data replicando el problema real
masked_text = """Antecedentes personales:
[**SEXO_SUJETO_ASISTENCIA**] de [**EDAD_SUJETO_ASISTENCIA**], alérgica al ácido acetilsalicílico.
Fumadora actual de 3 cig/día, previamente 2-3 paq/día.
ANTECEDENTES PATOLÓGICOS:"""

anonymized_text = """Mujer de [**49 años**], [**ALERGICA_SUJETO_ASISTENCIA**] al ácido acetilsalicílico.
Fumadora actual de [**3 cig/día**], previamente 2-3 paq/día.
ANTECEDENTES PATOLÓGICOS:"""

print("🔍 Testing alignment function...")
result = extract_tags_from_files(masked_text, anonymized_text)

print("\n📊 Results:")
for line_id, line_data in result.items():
    if line_id.startswith("line_"):
        print(f"\n{line_id}:")
        print(f"  textA: '{line_data['textA']}'")
        print(f"  textB: '{line_data['textB']}'")
        print(f"  labelA: {line_data['labelA']}")
        print(f"  labelB: {line_data['labelB']}")
        print(f"  tp={line_data['metrics']['tp']}, fp={line_data['metrics']['fp']}, fn={line_data['metrics']['fn']}")