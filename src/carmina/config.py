import os
from dotenv import load_dotenv

# Cargar variables de entorno una sola vez
load_dotenv()

# Configuración general
DEBUG_ENABLED = os.getenv('DEBUG_ENABLED', 'False').lower() in ('true', '1', 'yes')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Configuración de modelos
DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'claude-3.7-sonnet')
DEFAULT_TEMPERATURE = float(os.getenv('DEFAULT_TEMPERATURE', '0.7'))
DEFAULT_MAX_TOKENS = int(os.getenv('DEFAULT_MAX_TOKENS', '1000'))

# Configuración de carpetas
INPUT_DIR = os.getenv('INPUT_DIR', 'data/input.json')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'data/outputs/')
METRICS_DIR = os.getenv('METRICS_DIR', 'metrics/')
DEBUG_DIR = os.getenv('DEBUG_DIR', 'data/outputs/debug/')