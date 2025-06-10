# fca_lib

A Python library for Formal Concept Analysis (FCA), offering command‑line tools, core functionalities, and examples to get you started.

## 📦 Table of Contents

- [Installation](#installation)  
- [Requirements](#requirements)  
- [Usage](#usage)  
  - [Library Usage](#library-usage)  
  - [Command-line Interface (CLI)](#command-line-interface-cli)  
  - [Examples](#examples)  
- [Structure](#structure)  
- [Testing](#testing)  
- [Contributing](#contributing)  
- [License](#license)

---

## Installation

Install via `pip`:

```bash
pip install .
```

Or directly from source:

```bash
git clone https://github.com/Pequanta/fca_lib.git
cd fca_lib
pip install .
```

---

## Requirements

This project is compatible with Python 3.7+ and relies on the following dependencies:

Dependencies are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

To use a Python virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

Or with Poetry:

```bash
poetry install
```

---

## Usage

### Library Usage

```python
from fca import Context, ConceptLattice

# Load or construct a formal context
ctx = Context.from_csv('data/context.csv')

# Build and analyze the concept lattice
lattice = ConceptLattice(ctx)
for concept in lattice.concepts:
    print(concept)
```

*Adjust according to actual API in `fca/`.*

---

### Command-line Interface (CLI)

The CLI scripts are in the `cli/` folder. You can run them directly:

```bash
python -m fca_lib.cli.command_name --help
```

Or if installed as an entry-point:

```bash
fca-lib [command] [options]
```

Available commands include:

- `generate-lattice` – Generate concept lattices from context files  
- `export-diagram` – Export lattice visualizations  

*(Customize based on real CLI scripts.)*

---

### Examples

The `examples/` directory provides sample scripts:

```bash
python examples/example_build_lattice.py
```

These demonstrate:

- Building lattices from data  
- Exporting DOT/graph outputs  
- Annotating concepts with attributes

---

## Structure

```
fca_lib/
├── cli/         # Command-line interface scripts
├── docs/        # Documentation files
├── examples/    # Example usage scripts
├── fca/         # Core FCA modules (contexts, lattices, algorithms)
├── scripts/     # Utility or helper scripts
├── tests/       # Unit tests
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Testing

Run the test suite with:

```bash
pytest
```

Ensure that all tests in `tests/` pass before committing.

---

## Contributing

1. Fork the repo  
2. Create a new branch: `git checkout -b feature/my-feature`  
3. Make your changes and add tests  
4. Run tests and ensure formatting:  
   ```bash
   pytest
   black .
   ```
5. Submit a pull request

---

## License

This project is licensed under the **GNU General Public License v2.0**. See the [LICENSE](LICENSE) file for details.
