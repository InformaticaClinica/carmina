#!/usr/bin/env python3

import sys
sys.path.append('.')

from src.carmina.metrics.compare_line import extract_tags_from_files

print("🔢 Testing numbering alignment fix...")

# Test data with renumbered lists
masked_text = """8. BICITOPENIA (leucopenia-neutropenia y anemia macrocítica) desde [**FECHAS**] posiblemente en contexto de displasia asociada a VIH avanzado y déficit de factores madurativos, suplementado con ácido fólico. * Mielograma [**FECHAS**] compatible con déficit de factores de maduración vs mielodisplasia, sin atipias celulares, cultivos negativos.
9. PTOSIS OJO IZQUIERDO detectada en [**FECHAS**], no seguimiento.
10. OTRO DIAGNÓSTICO."""

anonymized_text = """7. BICITOPENIA (leucopenia-neutropenia y anemia macrocítica) desde 2009 posiblemente en contexto de displasia asociada a VIH avanzado y déficit de factores madurativos, suplementado con ácido fólico. * Mielograma mar/12 compatible con déficit de factores de maduración vs mielodisplasia, sin atipias celulares, cultivos negativos.
8. PTOSIS OJO IZQUIERDO detectada en mar/11, no seguimiento.
9. OTRO DIAGNÓSTICO."""

result = extract_tags_from_files(masked_text, anonymized_text)

print("\n📊 Numbering alignment results:")
for line_id, line_data in result.items():
    if line_id.startswith("line_"):
        print(f"\n{line_id}:")
        print(f"  textA: '{line_data['textA'][:60]}{'...' if len(line_data['textA']) > 60 else ''}'")
        print(f"  textB: '{line_data['textB'][:60]}{'...' if len(line_data['textB']) > 60 else ''}'")
        print(f"  labelA: {line_data['labelA']}")
        print(f"  labelB: {line_data['labelB']}")
        
        # Check if content matches despite different numbers
        textA_content = line_data['textA'].replace('[**FECHAS**]', '').strip()
        textB_content = line_data['textB'].strip()
        
        # Remove numbers from both
        import re
        textA_no_num = re.sub(r'^\d+\.\s*', '', textA_content).strip()
        textB_no_num = re.sub(r'^\d+\.\s*', '', textB_content).strip()
        
        if 'BICITOPENIA' in textA_content and 'BICITOPENIA' in textB_content:
            print(f"  ✅ BICITOPENIA correctly aligned despite number mismatch (8 vs 7)")
        elif 'PTOSIS' in textA_content and 'PTOSIS' in textB_content:
            print(f"  ✅ PTOSIS correctly aligned despite number mismatch (9 vs 8)")
        elif 'OTRO' in textA_content and 'OTRO' in textB_content:
            print(f"  ✅ OTRO correctly aligned despite number mismatch (10 vs 9)")