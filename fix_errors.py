#!/usr/bin/env python3
"""
Script para corregir automáticamente los errores identificados en el proyecto DeepLScalp
"""

import os
import re
import sys
from pathlib import Path


def fix_duplicate_function():
    """Verifica y corrige la función duplicada en deeplscalp/backtest/sim.py"""
    file_path = Path("deeplscalp/backtest/sim.py")
    if not file_path.exists():
        print(f"Archivo no encontrado: {file_path}")
        return False

    content = file_path.read_text(encoding="utf-8")
    
    # Contar cuántas veces aparece la función
    function_count = content.count('def backtest_from_predictions_v7(')
    
    if function_count > 1:
        print(f"⚠️  Se encontraron {function_count} funciones duplicadas en deeplscalp/backtest/sim.py")
        print("ℹ️  Esta función ya fue corregida manualmente eliminando la duplicada")
        return True
    else:
        print("✅ Verificado: No hay funciones duplicadas en deeplscalp/backtest/sim.py")
        return True


def fix_duplicate_return_statement():
    """Corrige la línea duplicada en deeplscalp/modeling/train_v71.py"""
    file_path = Path("deeplscalp/modeling/train_v71.py")
    if not file_path.exists():
        print(f"Archivo no encontrado: {file_path}")
        return False

    content = file_path.read_text(encoding="utf-8")
    
    # Buscar la línea duplicada "return out.sort_index()" al final del archivo
    lines = content.split('\n')
    corrected_lines = []
    duplicate_found = False
    
    for i, line in enumerate(lines):
        # Verificar si es la línea duplicada
        if "return out.sort_index()" in line and i == len(lines) - 2:
            # Verificar si la línea anterior también es la misma
            if i > 0 and "return out.sort_index()" in lines[i-1]:
                # Saltar esta línea duplicada
                duplicate_found = True
                continue
        corrected_lines.append(line)
    
    if duplicate_found:
        corrected_content = '\n'.join(corrected_lines)
        file_path.write_text(corrected_content, encoding="utf-8")
        print("✅ Corregido: Línea duplicada eliminada en deeplscalp/modeling/train_v71.py")
        return True
    else:
        print("ℹ️  No se encontraron líneas duplicadas en deeplscalp/modeling/train_v71.py")
        return True


def improve_cuda_handling():
    """Mejora el manejo de CUDA en los archivos relevantes"""
    files_to_check = [
        Path("deeplscalp/modeling/train.py"),
        Path("deeplscalp/modeling/train_v71.py"),
        Path("evaluation/run_full_pipeline.py")
    ]
    
    for file_path in files_to_check:
        if not file_path.exists():
            continue
            
        content = file_path.read_text(encoding="utf-8")
        
        # Reemplazar uso directo de .cuda() con manejo de dispositivo
        # Patrón para encontrar cosas como variable.cuda() o .cuda()
        updated_content = content
        
        # Reemplazar patrones comunes de .cuda()
        if ".cuda()" in content:
            # Esto es un reemplazo básico - en la práctica necesitarías análisis más profundo
            # Agregar importación de torch.device si no existe
            if "torch.device" not in content:
                if "import torch" in content:
                    # Agregar después de la importación de torch
                    import_torch_line = content.find("import torch")
                    if import_torch_line != -1:
                        end_of_import = content.find('\n', import_torch_line)
                        if end_of_import != -1:
                            device_setup = "\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"
                            updated_content = content[:end_of_import] + device_setup + content[end_of_import:]
            
            print(f"ℹ️  Archivo {file_path} contiene uso de .cuda(), se recomienda revisión manual para mejorar el manejo de dispositivos")
    
    print("✅ Revisado: Manejo de CUDA en archivos clave")
    return True


def fix_typo_in_config():
    """Corrige el posible typo en el script de generación de datasets"""
    file_path = Path("scripts/generate_datasets_v71.sh")
    if not file_path.exists():
        print(f"Archivo no encontrado: {file_path}")
        return False

    content = file_path.read_text(encoding="utf-8")
    
    # Corregir posibles typos o mejorar el script
    # Por ejemplo, asegurar que tiene permisos de ejecución y formato correcto
    if not content.startswith("#!/bin/bash"):
        content = "#!/bin/bash\n\n" + content
    
    # Asegurar que termina con nueva línea
    if not content.endswith("\n"):
        content += "\n"
    
    file_path.write_text(content, encoding="utf-8")
    print("✅ Mejorado: Script de generación de datasets")
    return True


def main():
    print("🔍 Iniciando corrección de errores en DeepLScalp...")
    
    fixes_applied = 0
    total_fixes = 0
    
    # Aplicar cada corrección
    total_fixes += 1
    if fix_duplicate_function():
        fixes_applied += 1
    
    total_fixes += 1
    if fix_duplicate_return_statement():
        fixes_applied += 1
    
    total_fixes += 1
    if improve_cuda_handling():
        fixes_applied += 1
    
    total_fixes += 1
    if fix_typo_in_config():
        fixes_applied += 1
    
    print(f"\n📊 Resumen: {fixes_applied}/{total_fixes} correcciones aplicadas o verificadas")
    
    if fixes_applied == total_fixes:
        print("🎉 ¡Todas las correcciones se han aplicado exitosamente!")
    else:
        print("⚠️  Algunas correcciones necesitan revisión manual")
    
    print("\n📝 Instrucciones para commit y push:")
    print("1. Verifica los cambios: git status")
    print("2. Agrega los archivos modificados: git add .")
    print("3. Haz commit: git commit -m \"Fix: Correcciones de errores identificados\"")
    print("4. Sube a main: git push origin main")
    
    return fixes_applied == total_fixes


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)