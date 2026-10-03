"""
Root runner for Plant Disease Detection Web Application
Allows running directly with `python run.py` from the project root.
"""
import sys
import os
from pathlib import Path

# Ensure UTF-8 output across all consoles
sys.stdout.reconfigure(encoding='utf-8')

# Add 'Flask Deployed App' to Python system path
app_dir = Path(__file__).resolve().parent / 'Flask Deployed App'
sys.path.insert(0, str(app_dir))

# Change working directory so relative assets work seamlessly
os.chdir(str(app_dir))

from app import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("=" * 65)
    print("  FLORASCAN AI - PLANT DISEASE DETECTION & HEALTH SYSTEM")
    print(f"  Web Application Running at: http://127.0.0.1:{port}")
    print("  REST API available at:     http://127.0.0.1:{port}/api/predict")
    print("=" * 65)
    app.run(host='0.0.0.0', port=port, debug=False)
