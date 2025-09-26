# FCA_LIB

A research-oriented Python library for **Formal Concept Analysis (FCA)** with extensions for **classification, hypothesis generation, and optimization**. The project combines classical FCA methods with experimental approaches such as the **John Stuart Mill (JSM) method**, **Titanic iceberg pruning**, and **QUBO-based optimization** (for both classical solvers and quantum-inspired workflows).

---

## 📂 Project Structure

* **`fca/`** – Core FCA implementations

  * `algorithms/` – Concept generation algorithms (Next Closure, Iceberg concepts).
  * `utils/` – Supporting utilities (bitset operations, tools, models, fuzzy logic).
  * `encoders.py` – Context encoding utilities.
  * `concept_lattice.py` – Lattice construction and concept handling.
  * `graph_representations.py` – Graph-based representations of lattices.
  * `qubo_formulation/` – QUBO construction and classical solver integration.

* **`scripts/examples/`** – Example workflows and experiments

  * `jsm_method/` – JSM application (data preprocessing, hypothesis generation, classification).
  * `run_example.py` – Entry point for running example workflows.
  * `placeholder_example.py` – Placeholder file for uncomment section usage.

* **`docs/`** – Documentation and supporting material.

---

## ⚙️ Features

* **Formal Concept Analysis (FCA)**

  * Next Closure algorithm for generating concepts.
  * Iceberg pruning (Titanic method) to manage concept explosion.
  * Graph and lattice representations of contexts.

* **JSM Method (Hypothesis-Based Classification)**

  * Hypothesis generation from positive and negative contexts.
  * Classification of undetermined examples using FCA-derived rules.
  * Handling of contradictory and insufficient data cases.

* **Optimization via QUBO**

  * QUBO formulation for rule selection and candidate pruning.
  * Simulated annealing as a baseline solver.
  * Integration prepared for quantum solvers.

* **Data Handling**

  * Bitset-based context encoding.
  * Preprocessing utilities for structured data.

---

## 🚀 Usage

### Installation

Clone the repository and install dependencies (requires Python 3.10+):

```bash
git clone https://github.com/Pequanta/fca_lib.git
cd fca_lib
```

if you have an api token for qci-client, create an .env file inside the fca directory

```bash
QCI_TOKEN=YOUR_TOKEN  
QCI_URL=https://api.qci-prod.com
```

 [!Only in the case of dirac token support] -> **Now make the value of the following variable in `scripts/examples/run_example.py` True ** to enable QCI-client integration and comment out the one following it.

```python
#scripts/examples/run_example.py
214 |    quantum_available = False
```

### Running Examples

Run the included example:

```bash
bash scripts/run_example.sh
```

---

## 🔮 Roadmap

* Extend QUBO integration to quantum solvers.
* Improve Titanic iceberg pruning for large-scale datasets.

---

## 📜 License

This project is under [MIT License](LICENSE).


