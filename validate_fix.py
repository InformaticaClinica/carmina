#!/usr/bin/env python3

import sys
sys.path.append('.')

from src.carmina.metrics.compare_line import extract_all_metrics

print("🔍 Validating alignment fix with real problematic data...")

# Real data from the debug file - exact problematic case
ground_truth_texts = [
    "\nAntecedentes personales:\n[**SEXO_SUJETO_ASISTENCIA**] de [**EDAD_SUJETO_ASISTENCIA**], alérgica al ácido acetilsalicílico, sulfamidas, sulfametoxazol/trimetoprim, amoxicilina/clavulánico (rash cutáneo) e hipersensibilidad al abacavir.\nFumadora actual de 3 cig/día, previamente 2-3 paq/día con dosis acumulada de 50 paq/año, no bebedora de alcohol. Consumidora esporádica de cocaína y metanfetamina inhalada, no UDVP. ndependiente para las actividades básicas de la vida diaria, vive en la calle en situación de indigencia.\nANTECEDENTES PATOLÓGICOS:"
]

prediction_texts = [
    "Mujer de [**49 años**], [**ALÉRGICA_SUJETO_ASISTENCIA**] al ácido acetilsalicílico, sulfamidas, sulfametoxazol/trimetoprim, amoxicilina/clavulánico (rash cutáneo) e hipersensibilidad al abacavir.\nFumadora actual de [**3 cig/día**], previamente 2-3 paq/día con dosis acumulada de 50 paq/año, no bebedora de alcohol. Consumidora esporádica de cocaína y metanfetamina inhalada, no [**UDVP**]. Independiente para las actividades básicas de la vida diaria, vive en [**CALLE**] en situación de [**INDIGENCIA**].\nANTECEDENTES PATOLÓGICOS:"
]

result = extract_all_metrics(
    ground_truth_identity_texts=[""],  # Not used for label mode
    prediction_identity_texts=[""],    # Not used for label mode  
    ground_truth_texts=ground_truth_texts,
    prediction_texts=prediction_texts,
    filenames=["CARMEN-I_IA_ANTECEDENTES_101.txt"],
    languages=["es"]
)

print("\n📊 Label mode results validation:")
label_result = result.get("files", [{}])[0].get("label", {})

empty_lines_found = 0
coordination_issues = 0

for key, value in label_result.items():
    if key.startswith("line_"):
        print(f"\n{key}:")
        print(f"  textA: '{value['textA'][:80]}{'...' if len(value['textA']) > 80 else ''}'")
        print(f"  textB: '{value['textB'][:80]}{'...' if len(value['textB']) > 80 else ''}'")
        print(f"  labelA: {value['labelA']}")
        print(f"  labelB: {value['labelB']}")
        
        # Check for problems
        if not value['textA'].strip() or not value['textB'].strip():
            empty_lines_found += 1
            print(f"  ⚠️  EMPTY LINE DETECTED!")
        
        # Check for obvious misalignment
        if 'antecedentes' in value['textA'].lower() and 'mujer' in value['textB'].lower():
            coordination_issues += 1
            print(f"  ⚠️  COORDINATION ISSUE!")

print(f"\n🔍 Validation Summary:")
print(f"   Empty lines found: {empty_lines_found}")
print(f"   Coordination issues: {coordination_issues}")

if empty_lines_found == 0 and coordination_issues == 0:
    print("✅ VALIDATION PASSED - No empty lines or coordination issues!")
else:
    print("❌ VALIDATION FAILED - Issues found!")

print(f"\n📈 Overall metrics:")
if 'metrics' in result and 'label' in result['metrics']:
    metrics = result['metrics']['label']
    print(f"   Precision: {metrics.get('precision', 0):.3f}")
    print(f"   Recall: {metrics.get('recall', 0):.3f}")
    print(f"   F1: {metrics.get('f1', 0):.3f}")