export PYTHONPATH=$PYTHONPATH:./fca

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "Running example for with FCA"
#!/bin/bash

if test -d venv; then
    echo "Virtual environment already exists."
    VENV_EXISTS=true
else
    echo "Creating virtual environment..."
    python3.12 -m venv venv
    VENV_EXISTS=false
fi

# Activate the virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "ERROR: Failed to create or find the virtual environment."
    exit 1
fi

# Use the variable correctly
if [ "$VENV_EXISTS" = false ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    echo "Virtual environment exists. Skipping installation."
fi


if python3.12 -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('tkinter') else 1)"; then
    echo "tkinter is already installed."
else
    echo "Installing tkinter..."
    sudo apt install -y python3.12-tk
fi
echo "Running example script..."
python3.12 scripts/examples/run_example.py
