"""
Script Principal Integrado - MLOps Pipeline
===========================================
Punto de entrada único para ejecutar el sistema completo:
1. Pipeline de entrenamiento
2. Dashboard de monitoreo

Autor: Alexis Jacquet
Proyecto: M5 - Henry - Avance 3
Fecha: Febrero 2026
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime


def print_banner():
    """Imprime banner del proyecto"""
    print("\n" + "="*80)
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "MLOps PIPELINE - PROYECTO M5" + " "*30 + "║")
    print("║" + " "*10 + "Sistema de Predicción de Pagos con Monitoreo" + " "*23 + "║")
    print("╚" + "="*78 + "╝")
    print("="*80)
    print(f"\n📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("👤 Autor: Alexis Jacquet")
    print("🎓 Programa: Henry - Data Science Bootcamp")
    print("="*80 + "\n")


def run_training_pipeline():
    """Ejecuta el pipeline de entrenamiento de modelos"""
    print("\n🚀 INICIANDO PIPELINE DE ENTRENAMIENTO")
    print("-" * 80)
    
    try:
        # Ejecutar pipeline
        subprocess.run([sys.executable, "run_pipeline.py"], check=True)
        
        print("\n" + "="*80)
        print("✅ PIPELINE DE ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("="*80)
        print("\n📁 Resultados guardados en: results/")
        print("   ✓ model_comparison.png")
        print("   ✓ roc_curves.png")
        print("   ✓ confusion_matrices.png")
        print("   ✓ evaluation_report.txt")
        print("   ✓ model_results.csv")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: El pipeline de entrenamiento falló")
        print(f"   Detalles: {e}")
        return False
    except FileNotFoundError:
        print(f"\n❌ ERROR: No se encontró el archivo run_pipeline.py")
        return False


def run_monitoring_dashboard():
    """Ejecuta el dashboard de monitoreo con Streamlit"""
    print("\n🎯 INICIANDO DASHBOARD DE MONITOREO")
    print("-" * 80)
    print("\n📊 Abriendo aplicación Streamlit...")
    print("🌐 URL: http://localhost:8501")
    print("\n⚠️  Presiona Ctrl+C para detener el servidor")
    print("-" * 80)
    
    try:
        # Ejecutar Streamlit
        subprocess.run(
            ["streamlit", "run", "app_streamlit.py"],
            check=True
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Dashboard detenido por el usuario")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: El dashboard falló")
        print(f"   Detalles: {e}")
        return False
    except FileNotFoundError:
        print(f"\n❌ ERROR: Streamlit no está instalado o app_streamlit.py no existe")
        print(f"   Instala con: pip install streamlit")
        return False
    
    return True


def check_dependencies():
    """Verifica que las dependencias estén instaladas"""
    print("🔍 Verificando dependencias...\n")
    
    dependencies = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',
        'streamlit': 'streamlit',
        'plotly': 'plotly',
        'scipy': 'scipy'
    }
    
    missing = []
    
    for name, import_name in dependencies.items():
        try:
            __import__(import_name)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} - NO INSTALADO")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️  Dependencias faltantes: {', '.join(missing)}")
        print("\n💡 Instala con: pip install -r requirements.txt")
        return False
    
    print("\n✅ Todas las dependencias están instaladas\n")
    return True


def show_menu():
    """Muestra menú interactivo"""
    print("\n" + "="*80)
    print("MENÚ PRINCIPAL")
    print("="*80)
    print("\n1. 🚀 Ejecutar Pipeline de Entrenamiento")
    print("2. 🎯 Abrir Dashboard de Monitoreo")
    print("3. 🔄 Ejecutar Pipeline + Dashboard")
    print("4. 🔍 Verificar Dependencias")
    print("5. 📚 Mostrar Ayuda")
    print("6. ❌ Salir")
    print("\n" + "="*80)
    
    choice = input("\nSelecciona una opción (1-6): ").strip()
    return choice


def show_help():
    """Muestra información de ayuda"""
    print("\n" + "="*80)
    print("AYUDA - MLOps Pipeline")
    print("="*80)
    
    print("""
📖 DESCRIPCIÓN:
   Sistema completo de MLOps para predicción de pagos con monitoreo de data drift.

🚀 USO RÁPIDO:
   
   1. Instalar dependencias:
      pip install -r requirements.txt
   
   2. Ejecutar pipeline de entrenamiento:
      python main.py --train
   
   3. Abrir dashboard de monitoreo:
      python main.py --dashboard
   
   4. Ejecutar todo:
      python main.py --all

🎯 MODOS DE EJECUCIÓN:
   
   --train, -t        Ejecuta solo el pipeline de entrenamiento
   --dashboard, -d    Abre solo el dashboard de monitoreo
   --all, -a          Ejecuta pipeline y luego abre dashboard
   --check, -c        Verifica dependencias instaladas
   --interactive, -i  Modo interactivo con menú (por defecto)

📁 ESTRUCTURA DE SALIDA:
   
   results/
   ├── model_comparison.png       - Comparación de modelos
   ├── roc_curves.png             - Curvas ROC
   ├── confusion_matrices.png     - Matrices de confusión
   ├── evaluation_report.txt      - Reporte detallado
   ├── model_results.csv          - Resultados en CSV
   └── monitoring/                - Reportes de drift
       └── drift_report_*.json

📊 DASHBOARD:
   
   El dashboard incluye:
   - Detección de data drift (KS, PSI, JS, Chi²)
   - Sistema de alertas automáticas
   - Visualizaciones interactivas
   - Análisis temporal
   - Recomendaciones

🔗 MÁS INFORMACIÓN:
   
   Ver README_AVANCE3.md para documentación completa

    """)


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description='MLOps Pipeline - Sistema Integrado de Entrenamiento y Monitoreo'
    )
    
    parser.add_argument(
        '--train', '-t',
        action='store_true',
        help='Ejecutar pipeline de entrenamiento'
    )
    
    parser.add_argument(
        '--dashboard', '-d',
        action='store_true',
        help='Abrir dashboard de monitoreo'
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Ejecutar pipeline y dashboard'
    )
    
    parser.add_argument(
        '--check', '-c',
        action='store_true',
        help='Verificar dependencias'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Modo interactivo con menú'
    )
    
    args = parser.parse_args()
    
    # Si no hay argumentos, ejecutar en modo interactivo
    if not any([args.train, args.dashboard, args.all, args.check, args.interactive]):
        args.interactive = True
    
    print_banner()
    
    # Modo de verificación
    if args.check:
        check_dependencies()
        return
    
    # Modo interactivo
    if args.interactive:
        while True:
            choice = show_menu()
            
            if choice == '1':
                if check_dependencies():
                    run_training_pipeline()
                    input("\nPresiona Enter para continuar...")
            
            elif choice == '2':
                if check_dependencies():
                    run_monitoring_dashboard()
            
            elif choice == '3':
                if check_dependencies():
                    success = run_training_pipeline()
                    if success:
                        input("\n✅ Pipeline completado. Presiona Enter para abrir dashboard...")
                        run_monitoring_dashboard()
            
            elif choice == '4':
                check_dependencies()
                input("\nPresiona Enter para continuar...")
            
            elif choice == '5':
                show_help()
                input("\nPresiona Enter para continuar...")
            
            elif choice == '6':
                print("\n👋 ¡Hasta luego!")
                break
            
            else:
                print("\n❌ Opción inválida. Por favor selecciona 1-6.")
                input("Presiona Enter para continuar...")
    
    # Modo comando
    else:
        if not check_dependencies():
            sys.exit(1)
        
        if args.all:
            success = run_training_pipeline()
            if success:
                print("\n✅ Abriendo dashboard...")
                run_monitoring_dashboard()
        
        elif args.train:
            run_training_pipeline()
        
        elif args.dashboard:
            run_monitoring_dashboard()


if __name__ == "__main__":
    main()
