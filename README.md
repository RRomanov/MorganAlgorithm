# MorganAlgorithm
Python implementation of the Morgan algorithm with SMILES input, atom numbering, iterations and molecular graph images.
# Morgan Algorithm — Atom Numbering Tool

A simple Python script that applies the classical **Morgan Algorithm** to number atoms in a molecule based on their topological environment.

This tool takes a SMILES string, computes atom invariants through Morgan iterations, assigns final numbering, and saves images + a text report.

---

## Features

* Input a SMILES string
* Automatically determines compound name (PubChem)
* Computes Morgan invariants step‑by‑step
* Generates images of each iteration
* Produces final atom numbering
* Saves results to `MorganResults/`

---

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd MorganAlgorithm
pip install rdkit-pypi pubchempy pillow
```

If you use Conda:

```bash
conda install -c conda-forge rdkit
```

---

## Usage

Run the script:

```bash
python MorganAlgorithm.py
```

Enter a SMILES string when prompted:

```
Enter molecule SMILES:
```

Example:

```
c1ccccc1
```

Outputs will appear in:

```
MorganResults/
```

---

## Output Files

* `iteration_#.png` — Morgan iteration steps
* `final_numbering.png` — atom numbering result
* `final_report.txt` — text summary
* `molecule_name.txt` — detected compound name

---

## Citing

```
Romanov Ruslan A. (https://github.com/RRomanov), Isaev Yaroslav I. (https://github.com/IsaevYaroslavIv)
                      romanovnsx@gmail.com                           isaev.yaroslav.ivanovo@gmail.com
Kovanova Mariia A., Ph.D. of Chemical Sciences, Associate Professor (mariia.a.kovanova@gmail.com)
Morgan Algorithm — Atom Numbering Tool. GitHub.
```

---

## License

MIT License.
