import os
import sys

# Ensure the fca directory is accessible throughout the tests package
fca_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../fca'))
if fca_path not in sys.path:
    sys.path.insert(0, fca_path)