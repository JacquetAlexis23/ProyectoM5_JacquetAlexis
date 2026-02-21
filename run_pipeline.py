"""
Pipeline Principal - Integración Completa MLOps
===============================================
Script principal que ejecuta el pipeline completo de:
1. Feature Engineering (v1.1.0)
2. Model Training & Evaluation (v1.0.1)
3. Generación de reportes y visualizaciones

Autor: Alexis Jacquet
Proyecto: M5 - Henry
Fecha: Febrero 2026
"""

import sys
import os

# Añadir el directorio src al path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'mlops_pipeline', 'src'))

from ft_engineering import load_and_prepare_data
from model_training_evaluation import main_training_pipeline
import warnings
warnings.filterwarnings('ignore')


def main():
    """Ejecuta el pipeline completo de MLOps"""
    
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "PIPELINE MLOPS - PROYECTO M5" + " "*30 + "║")
    print("║" + " "*15 + "Feature Engineering + Model Training" + " "*26 + "║")
    print("╚" + "="*78 + "╝")
    
    # Configuración - Trabajar desde el directorio del script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = script_dir
    DATA_PATH = os.path.join(project_root, "data", "Base_de_datos.csv")
    OUTPUT_DIR = os.path.join(project_root, "results/")
    
    try:
        # ==================================================================
        # FASE 1: FEATURE ENGINEERING (v1.1.0)
        # ==================================================================
        print("\n" + "▶"*40)
        print("FASE 1: FEATURE ENGINEERING - Versión 1.1.0")
        print("▶"*40)
        
        data = load_and_prepare_data(DATA_PATH, test_size=0.2, random_state=42)
        
        print("\n✅ Feature Engineering completado exitosamente!")
        print(f"   • Features generadas: {data['X_train'].shape[1]}")
        print(f"   • Muestras de entrenamiento: {data['X_train'].shape[0]}")
        print(f"   • Muestras de prueba: {data['X_test'].shape[0]}")
        
        # ==================================================================
        # FASE 2: MODEL TRAINING & EVALUATION (v1.0.1)
        # ==================================================================
        print("\n" + "▶"*40)
        print("FASE 2: MODEL TRAINING & EVALUATION - Versión 1.0.1")
        print("▶"*40)
        
        models_dict, results_df, best_model_name = main_training_pipeline(
            data['X_train'], 
            data['y_train'],
            data['X_test'], 
            data['y_test'],
            output_dir=OUTPUT_DIR
        )
        
        # ==================================================================
        # RESUMEN FINAL
        # ==================================================================
        print("\n" + "╔" + "="*78 + "╗")
        print("║" + " "*25 + "RESUMEN FINAL DEL PIPELINE" + " "*27 + "║")
        print("╚" + "="*78 + "╝")
        
        print(f"\n🎯 MEJOR MODELO SELECCIONADO: {best_model_name}")
        print(f"\n📊 Métricas del mejor modelo:")
        best_results = results_df.iloc[0]
        print(f"   • ROC-AUC:    {best_results['ROC-AUC']:.4f}")
        print(f"   • F1-Score:   {best_results['F1-Score']:.4f}")
        print(f"   • Precision:  {best_results['Precision']:.4f}")
        print(f"   • Recall:     {best_results['Recall']:.4f}")
        print(f"   • Accuracy:   {best_results['Accuracy']:.4f}")
        print(f"   • Tiempo:     {best_results['Training_Time']:.2f}s")
        
        print(f"\n📁 Archivos generados en '{OUTPUT_DIR}':")
        print(f"   ✓ model_comparison.png - Comparación visual de todos los modelos")
        print(f"   ✓ roc_curves.png - Curvas ROC de los mejores modelos")
        print(f"   ✓ confusion_matrices.png - Matrices de confusión")
        print(f"   ✓ evaluation_report.txt - Reporte detallado de evaluación")
        print(f"   ✓ model_results.csv - Tabla de resultados completa")
        
        print("\n" + "="*80)
        print("🎉 PIPELINE COMPLETADO EXITOSAMENTE!")
        print("="*80)
        
        # Top 3 modelos
        print("\n🏆 TOP 3 MODELOS:")
        for idx, row in results_df.head(3).iterrows():
            print(f"\n   {idx+1}. {row['Model']}")
            print(f"      ROC-AUC: {row['ROC-AUC']:.4f} | F1: {row['F1-Score']:.4f} | Tiempo: {row['Training_Time']:.2f}s")
        
        print("\n💡 PRÓXIMOS PASOS:")
        print("   1. Revisar las visualizaciones generadas en la carpeta 'results/'")
        print("   2. Analizar el reporte de evaluación detallado")
        print("   3. Considerar optimización de hiperparámetros del mejor modelo")
        print("   4. Implementar el modelo seleccionado en producción")
        print("   5. Configurar monitoreo de performance")
        
        print("\n" + "="*80 + "\n")
        
        return {
            'data': data,
            'models': models_dict,
            'results': results_df,
            'best_model': best_model_name
        }
        
    except FileNotFoundError:
        print(f"\n❌ ERROR: No se encontró el archivo '{DATA_PATH}'")
        print(f"   Asegúrate de que el archivo existe en el directorio actual.")
        return None
    
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = main()
    
    if result:
        print("✅ Pipeline ejecutado correctamente")
        print(f"✅ Mejor modelo: {result['best_model']}")
    else:
        print("❌ El pipeline falló. Revisa los errores anteriores.")
