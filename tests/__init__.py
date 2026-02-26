"""
Tests Package Initialization
=============================
Configuración del paquete de tests

Autor: Alexis Jacquet
HENRY M5 - Avance 3 Extra Credit
"""

import sys
from pathlib import Path

# Añadir rutas al path para imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "mlops_pipeline" / "src"))
