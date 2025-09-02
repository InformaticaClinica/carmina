#!/usr/bin/env python3

# Test directo de la función con datos reales
import sys
sys.path.append('.')

from src.carmina.metrics.compare_line import extract_all_metrics

# Datos reales del problema
ground_truth_texts = [
    "\nAntecedentes personales:\n[**SEXO_SUJETO_ASISTENCIA**] de [**EDAD_SUJETO_ASISTENCIA**], alérgica al ácido acetilsalicílico"
]

prediction_texts = [
    "Mujer de [**49 años**], [**ALERGICA_SUJETO_ASISTENCIA**] al ácido acetilsalicílico"
]

print("🧪 Testing with real data from the problematic case...")

result = extract_all_metrics(
    ground_truth_identity_texts=[""],  # No usado para label mode
    prediction_identity_texts=[""],    # No usado para label mode  
    ground_truth_texts=ground_truth_texts,
    prediction_texts=prediction_texts,
    filenames=["test.txt"],
    languages=["es"]
)

print("\n📊 Label mode results:")
label_result = result.get("files", [{}])[0].get("label", {})
for key, value in label_result.items():
    if key.startswith("line_"):
        print(f"\n{key}:")
        print(f"  textA: '{value['textA']}'")
        print(f"  textB: '{value['textB']}'")
        print(f"  labelA: {value['labelA']}")
        print(f"  labelB: {value['labelB']}")