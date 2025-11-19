from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import MolDraw2DCairo
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import io
import pubchempy as pcp
import os
from rdkit.Chem import rdMolDescriptors

# ===============================================================
#  PROJECT:     Implementation of the Morgan Algorithm
#  PURPOSE:     Automatic numbering of molecule atoms based on SMILES
# ---------------------------------------------------------------
#  AUTHOR:      Romanov Ruslan A. (https://github.com/RRomanov), Isaev Yaroslav I. (https://github.com/IsaevYaroslavIv) 
#                                      romanovnsx@gmail.com                          isaev.yaroslav.ivanovo@gmail.com
#  UNIVERSITY:  Ivanovo State University of Chemistry and Technology
#  SUPERVISOR:  Kovanova Mariia A., Ph.D. of Chemical Sciences, Associate Professor (mariia.a.kovanova@gmail.com)
#  YEAR:        2025
#  FILE:        MorganAlgorithm.py
# ---------------------------------------------------------------
#  DESCRIPTION:
#  This code implements the classical Morgan algorithm for
#  numbering atoms in molecules based on their topological
#  equivalence (Extended Connectivity)
#  The program performs the following actions:
#    • Accepts a molecule SMILES string
#    • Determines the compound name via PubChem
#    • Performs EC(0)...EC(n) iterations
#    • Carries out BFS atom numbering
#    • Generates visualizations of each step and the final numbering
#    • Saves all results in the /MorganResults directory
# ===============================================================

class CompactMorgan:
    def __init__(self, mol, name="molecule", output_dir="."):
        self.mol = mol
        self.name = name
        self.output_dir = output_dir
        self.heavy_atoms = [i for i in range(mol.GetNumAtoms())
                            if mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
        self.ec_history = []

    def calculate_initial_degrees(self):
        return {i: len([nbr for nbr in self.mol.GetAtomWithIdx(i).GetNeighbors()
                       if nbr.GetAtomicNum() > 1]) for i in self.heavy_atoms}

    def calculate_ec(self, previous_ec):
        return {i: sum(previous_ec[nbr.GetIdx()] for nbr in self.mol.GetAtomWithIdx(i).GetNeighbors()
                      if nbr.GetAtomicNum() > 1) for i in self.heavy_atoms}

    def get_unique_ec_count(self, ec_dict):
        """Returns the number of unique EC values"""
        return len(set(ec_dict.values()))

    def has_unique_maximum(self, ec_dict):
        """Checks whether there is a unique atom with the maximum value EC"""
        if not ec_dict:
            return False
        max_ec = max(ec_dict.values())
        candidates = [idx for idx, ec in ec_dict.items() if ec == max_ec]

        if len(candidates) == 1:
            atom_symbol = self.mol.GetAtomWithIdx(candidates[0]).GetSymbol()
            print(f"  Unique maximum found:” {atom_symbol}{candidates[0]} (EC={max_ec})")
            return True
        else:
            print(f"  Several atoms share the maximum value EC={max_ec}: {candidates}")
            return False

    def should_stop_iterations(self, current_ec, iteration):
        """
        Determines when to stop the iterations.
        - Minimum of 4 iterations
        - After the 4th: if the number of EC values is the same for iterations 3 and 4 AND there is a unique maximum vertex — stop
        - If not, continue until stabilization (±1 EC)
        """
        if iteration < 4:
            return False

        # Retrieve the history of unique counts
        unique_counts = [self.get_unique_ec_count(ec) for _, ec in self.ec_history]
        current_unique = self.get_unique_ec_count(current_ec)

        print(f"  # Retrieve the history of unique EC values: {current_unique}")

        # Check after the 4th iteration
        if iteration == 4:
            if unique_counts[2] == unique_counts[3] and self.has_unique_maximum(current_ec):
                print("✓ The stopping condition at the 4th iteration is met")
                return True
            else:
                print("➤ Continuing iterations (stopping conditions not met)")
                return False

        # After the 4th iteration, we check for stabilization within ±1 EC
        if iteration > 4:
            # Take the last 3 values to check for stabilization
            recent_counts = unique_counts[-3:]
            max_diff = max(recent_counts) - min(recent_counts)

            if max_diff <= 1 and self.has_unique_maximum(current_ec):
                print(f"✓ Stabilization within ±1 EC achieved: {recent_counts}")
                return True

        # Maximum number of iterations as a fallback safeguard
        if iteration >= 20:
            print("⚠ Maximum number of iterations reached")
            return True

        return False

    def get_atomic_priority(self, atom):
        priority_map = {6: 1, 7: 2, 8: 3, 16: 4, 15: 5, 9: 6, 17: 7}
        isotope = atom.GetIsotope()
        return (priority_map.get(atom.GetAtomicNum(), 0),
                -isotope if isotope != 0 else 0,
                atom.GetFormalCharge())

    def get_bond_priority(self, bond):
        """Bond priority: single > double > triple"""
        bond_priority = {
            Chem.rdchem.BondType.SINGLE: 3,    # higher-priority bond
            Chem.rdchem.BondType.DOUBLE: 2,    # medium-priority bond
            Chem.rdchem.BondType.TRIPLE: 1,    # lower-priority bond
            Chem.rdchem.BondType.AROMATIC: 3,  # like a single bond
        }
        return bond_priority.get(bond.GetBondType(), 0)

    def compare_atoms_by_rules(self, atom1_idx, atom2_idx, parent_atom_idx, final_ec):
        atom1, atom2 = self.mol.GetAtomWithIdx(atom1_idx), self.mol.GetAtomWithIdx(atom2_idx)

        print(f"    Comparison {atom1.GetSymbol()}{atom1_idx} and {atom2.GetSymbol()}{atom2_idx}:")

        # Rule 1: By EC
        ec1, ec2 = final_ec[atom1_idx], final_ec[atom2_idx]
        if ec1 != ec2:
            result = atom1_idx if ec1 > ec2 else atom2_idx
            print(f"      → Based on EC: {ec1} {'>' if ec1 > ec2 else '<'} {ec2}")
            return result

        print(f"      EC are identical: {ec1}")

        # Rule 2: By atom type
        prio1, prio2 = self.get_atomic_priority(atom1), self.get_atomic_priority(atom2)
        if prio1 != prio2:
            result = atom1_idx if prio1 > prio2 else atom2_idx
            print(f"      → By atom type: {atom1.GetSymbol()}{atom1_idx} {'>' if prio1 > prio2 else '<'} {atom2.GetSymbol()}{atom2_idx}")
            return result

        print(f"      Atom type is identical: {atom1.GetSymbol()}")

        # Rule 3: By isotopes
        iso1, iso2 = atom1.GetIsotope(), atom2.GetIsotope()
        if iso1 != iso2 and iso1 != 0 and iso2 != 0:
            result = atom1_idx if iso1 < iso2 else atom2_idx
            print(f"      → By isotope: {iso1} {'<' if iso1 < iso2 else '>'} {iso2}")
            return result

        # Rule 4: By charge
        charge1, charge2 = atom1.GetFormalCharge(), atom2.GetFormalCharge()
        if charge1 != charge2:
            result = atom1_idx if charge1 > charge2 else atom2_idx
            print(f"      → By charge: {charge1} {'>' if charge1 > charge2 else '<'} {charge2}")
            return result

        # Rule 5: By the bond type to the parent
        if parent_atom_idx is not None:
            bond1 = self.mol.GetBondBetweenAtoms(parent_atom_idx, atom1_idx)
            bond2 = self.mol.GetBondBetweenAtoms(parent_atom_idx, atom2_idx)

            if bond1 and bond2:
                bond_prio1 = self.get_bond_priority(bond1)
                bond_prio2 = self.get_bond_priority(bond2)

                if bond_prio1 != bond_prio2:
                    # Single (3) > double (2) > triple (1)
                    result = atom1_idx if bond_prio1 > bond_prio2 else atom2_idx
                    bond_types = {3: "singe", 2: "double", 1: "triple"}
                    type1 = bond_types.get(bond_prio1, bond_prio1)
                    type2 = bond_types.get(bond_prio2, bond_prio2)

                    # Determine which bond type is preferred
                    better_type = type1 if bond_prio1 > bond_prio2 else type2
                    worse_type = type2 if bond_prio1 > bond_prio2 else type1
                    print(f"      → By bond type: {better_type} > {worse_type}")
                    return result

        # If all rules are equal
        print(f"      → All rules are equal, selecting {atom1.GetSymbol()}{atom1_idx}")
        return atom1_idx

    def resolve_ties(self, atoms, parent_idx, final_ec):
        if len(atoms) == 1:
            return atoms[0]

        print(f"  Conflict resolution for atoms: {atoms}")
        remaining = atoms.copy()

        while len(remaining) > 1:
            better = self.compare_atoms_by_rules(remaining[0], remaining[1], parent_idx, final_ec)
            worse = remaining[1] if better == remaining[0] else remaining[0]
            atom_symbol = self.mol.GetAtomWithIdx(worse).GetSymbol()
            print(f"    Removed {atom_symbol}{worse}")
            remaining.remove(worse)

        result = remaining[0]
        atom_symbol = self.mol.GetAtomWithIdx(result).GetSymbol()
        print(f"  Selected atom: {atom_symbol}{result}")
        return result

    def number_atoms_bfs(self, start_atom, final_ec):
        numbering, used, current_num = {}, set(), 1
        numbering[start_atom] = current_num
        used.add(start_atom)
        atom_symbol = self.mol.GetAtomWithIdx(start_atom).GetSymbol()
        print(f"  Atom {atom_symbol}{start_atom} receives number 1")
        current_num += 1

        queue = [start_atom]

        while queue and current_num <= len(self.heavy_atoms):
            current = queue.pop(0)
            current_atom = self.mol.GetAtomWithIdx(current)

            print(f"\n  Processing the neighbors of the atom {current_atom.GetSymbol()}{current} (number {numbering[current]}):")

            neighbors = [nbr for nbr in current_atom.GetNeighbors()
                        if nbr.GetAtomicNum() > 1 and nbr.GetIdx() not in used]

            if not neighbors:
                print(f"    All neighbors are already numbered")
                continue

            # Group by EC
            ec_groups = defaultdict(list)
            for nbr in neighbors:
                ec_groups[final_ec[nbr.GetIdx()]].append(nbr.GetIdx())

            # Sort groups by EC (descending)
            for ec_value in sorted(ec_groups.keys(), reverse=True):
                atoms = ec_groups[ec_value]

                if len(atoms) == 1:
                    atom_idx = atoms[0]
                    numbering[atom_idx] = current_num
                    used.add(atom_idx)
                    atom_symbol = self.mol.GetAtomWithIdx(atom_idx).GetSymbol()
                    print(f"    Atom {atom_symbol}{atom_idx} (EC={ec_value}) receives number {current_num}")
                    queue.append(atom_idx)
                    current_num += 1
                else:
                    print(f"    Conflict: atoms {atoms} have the same EC={ec_value}")
                    remaining_atoms = atoms.copy()

                    while remaining_atoms:
                        next_atom = self.resolve_ties(remaining_atoms, current, final_ec)
                        numbering[next_atom] = current_num
                        used.add(next_atom)
                        atom_symbol = self.mol.GetAtomWithIdx(next_atom).GetSymbol()
                        print(f"    Atom {atom_symbol}{next_atom} receives number {current_num}")
                        queue.append(next_atom)
                        current_num += 1
                        remaining_atoms.remove(next_atom)

        return numbering

    def run_morgan_algorithm(self):
        print("=" * 50)
        print("CLASSICAL MORGAN ALGORITHM")
        print("=" * 50)

        # Stage EC(0): initial atom degrees
        ec_current = self.calculate_initial_degrees() 
        self.ec_history.append(("EC(0)", ec_current.copy()))
        print("\nSTEP 1: Initial atom degrees (EC₀)")
        self.print_ec_values(ec_current, "EC₀")

        # EC iterations
        iteration = 1
        while True:
            ec_prev = ec_current
            ec_current = self.calculate_ec(ec_prev)
            self.ec_history.append((f"EC({iteration})", ec_current.copy()))

            print(f"\nIteration {iteration}:")
            self.print_ec_values(ec_current, f"EC({iteration})")

            # Checking the stopping condition
            if self.should_stop_iterations(ec_current, iteration):
                break

            iteration += 1

        return ec_current

    def complete_numbering(self, final_ec):
        print("\n" + "=" * 50)
        print("FULL ATOM NUMBERING")
        print("=" * 50)

        # Finding the starting atom
        max_ec = max(final_ec.values())
        candidates = [idx for idx, ec in final_ec.items() if ec == max_ec]

        if len(candidates) == 1:
            start_atom = candidates[0]
        else:
            start_atom = self.resolve_ties(candidates, None, final_ec)

        atom_symbol = self.mol.GetAtomWithIdx(start_atom).GetSymbol()
        print(f"Starting the numbering from atom {atom_symbol}{start_atom} (EC={max_ec})")

        # Numbering
        final_numbering = self.number_atoms_bfs(start_atom, final_ec)

        # Checking all atoms
        if len(final_numbering) != len(self.heavy_atoms):
            missing = [idx for idx in self.heavy_atoms if idx not in final_numbering]
            for atom_idx in missing:
                next_num = len(final_numbering) + 1
                final_numbering[atom_idx] = next_num
                atom_symbol = self.mol.GetAtomWithIdx(atom_idx).GetSymbol()
                print(f"  Added atom {atom_symbol}{atom_idx} with number {next_num}")

        # Output of the final numbering
        print("\nFINAL NUMBERING:")
        print("Rank | Atom | Index | EC")
        print("-" * 30)

        for atom_idx, rank in sorted(final_numbering.items(), key=lambda x: x[1]):
            atom = self.mol.GetAtomWithIdx(atom_idx)
            print(f"{rank:4} | {atom.GetSymbol():4} | {atom_idx:6} | {final_ec[atom_idx]:2}")

        return final_numbering

    def print_ec_values(self, ec_dict, title):
        items = [f"{self.mol.GetAtomWithIdx(idx).GetSymbol()}{idx}:{val}"
                for idx, val in sorted(ec_dict.items())]
        print(f"{title}: {' | '.join(items)}")

    def visualize_all_steps(self):
        """Visualization of all algorithm steps in a textbook style"""
        print("\nCREATING VISUALIZATIONS...")

        for i, (title, ec_dict) in enumerate(self.ec_history):
            mol_viz = Chem.Mol(self.mol)

            # Adding EC labels to atoms
            for atom in mol_viz.GetAtoms():
                if atom.GetIdx() in ec_dict:
                    atom.SetProp('atomNote', str(ec_dict[atom.GetIdx()]))

            # Drawing the molecule with RDKit
            drawer = MolDraw2DCairo(600, 400)
            drawer.DrawMolecule(mol_viz)
            drawer.FinishDrawing()

             # Loading the image into Pillow
            png_data = drawer.GetDrawingText()
            base_img = Image.open(io.BytesIO(png_data)).convert("RGB")

            # --- Forming the label as in a textbook ---
            unique_vals = sorted(set(ec_dict.values()))
            n_unique = len(unique_vals)
            iter_label = title  # Now the iteration names are already in the correct format (EC(0), EC(1), …)
            text = f"{iter_label}: {n_unique} unique ({', '.join(map(str, unique_vals))})"

            # --- Adding a caption below the image ---
            padding = 60
            new_img = Image.new("RGB", (base_img.width, base_img.height + padding), "white")
            new_img.paste(base_img, (0, 0))

            draw = ImageDraw.Draw(new_img)
            try:
                font = ImageFont.truetype("arial.ttf", 18)
            except:
                font = ImageFont.load_default()

            bbox = draw.textbbox((0, 0), text, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            text_x = (new_img.width - text_w) // 2
            text_y = base_img.height + 10
            draw.text((text_x, text_y), text, fill="black", font=font)

            # --- Saving the file ---
            filename = os.path.join(self.output_dir, f"{self.name}_EC_{i}.png")
            new_img.save(filename)
            print(f"File created: {filename} — {text}")

    def visualize_final_numbering(self, final_numbering):
        """Visualization of the final numbering"""
        mol_viz = Chem.Mol(self.mol)

        for atom in mol_viz.GetAtoms():
            if atom.GetIdx() in final_numbering:
                atom.SetProp('atomNote', f"{final_numbering[atom.GetIdx()]}")

        # Drawing the molecule
        drawer = MolDraw2DCairo(600, 400)
        drawer.DrawMolecule(mol_viz)
        drawer.FinishDrawing()

        # Converting to a Pillow image
        png_data = drawer.GetDrawingText()
        base_img = Image.open(io.BytesIO(png_data)).convert("RGB")

        # Adding a caption at the bottom
        caption = "Final atom numbering"
        padding = 60
        new_img = Image.new("RGB", (base_img.width, base_img.height + padding), "white")
        new_img.paste(base_img, (0, 0))

        draw = ImageDraw.Draw(new_img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), caption, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        text_x = (new_img.width - text_w) // 2
        text_y = base_img.height + 10
        draw.text((text_x, text_y), caption, fill="black", font=font)

        # Saving the final file
        filename = os.path.join(self.output_dir, f"{self.name}_final.png")
        new_img.save(filename)
        print(f"File created: {filename} — {caption}")

# --- Universal analysis function ---
def analyze_molecule():
    """Analysis of any molecule based on the input SMILES"""
    smiles = input("Enter the molecule SMILES: ").strip()
    if not smiles:
        print("Error: SMILES not provided!")
        return

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("Error: invalid SMILES!")
        return

    # --- Retrieving the short name via PubChem---
    try:
        comps = pcp.get_compounds(smiles, 'smiles')
        if comps:
            comp = comps[0]
            if comp.synonyms and len(comp.synonyms) > 0:
                # Taking the short name (usually the first in the list)
                name = comp.synonyms[0].replace(" ", "_").replace("-", "_")
            elif comp.iupac_name:
                # If no synonyms are available, use the IUPAC name
                name = comp.iupac_name.replace(" ", "_").replace("-", "_")
            else:
                # If nothing is available — use the formula
                name = rdMolDescriptors.CalcMolFormula(mol)
        else:
            name = rdMolDescriptors.CalcMolFormula(mol)
    except Exception:
        print("⚠️  No PubChem compound found — using the formula instead of the name.")
        name = rdMolDescriptors.CalcMolFormula(mol)

    print(f"Name determined: {name}")
    print(f"\nMOLECULE: {name}")
    print(f"SMILES: {smiles}")
    print(f"Heavy atoms: {mol.GetNumHeavyAtoms()}\n")

    # --- Creating a folder for the results ---
    output_dir = "MorganResults"
    os.makedirs(output_dir, exist_ok=True)  # If the folder already exists, no error is raised
    print(f"The results will be saved to the folder: {output_dir}\n")

    morgan = CompactMorgan(mol, name, output_dir)
    final_ec = morgan.run_morgan_algorithm()

    # Visualization of all steps
    morgan.visualize_all_steps()

    # Final numbering
    final_numbering = morgan.complete_numbering(final_ec)
    morgan.visualize_final_numbering(final_numbering)
    save_text_report(name, smiles, mol, final_ec, final_numbering, morgan, output_dir)

    print(f"\nFINAL REPORT: atoms numbered {len(final_numbering)}")
    print("\nCreated files:")
    for i in range(len(morgan.ec_history)):
        print(f"- {name}_EC_{i}.png")
    print("- tyrosine_final.png")  # The name can also be adjusted later if you want

#  Creating a text report
from datetime import datetime  # Make sure that the import is already present at the top of the file

def save_text_report(name, smiles, mol, final_ec, final_numbering, morgan, output_dir):
    """Creates a text report based on the results of the Morgan algorithm"""
    path = os.path.join(output_dir, f"{name}_report.txt")

    # current date and time
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(path, "w", encoding="utf-8") as f:
         # --- Author section ---
        f.write(" AUTHOR SECTION:\n")
        f.write("  Author: Romanov Ruslan A. (https://github.com/RRomanov ), Isaev Yaroslav I. (https://github.com/IsaevYaroslavIv)\n")
        f.write("                                 romanovnsx@gmail.com                            isaev.yaroslav.ivanovo@gmail.com\n")
        f.write("  Supervisor: Kovanova Mariia A., Ph.D. of Chemical Sciences, Associate Professor (mariia.a.kovanova@gmail.com)\n")
        f.write("  University: Ivanovo State University of Chemistry and Technology\n")
        f.write("  Year: 2025\n")
        f.write("=" * 70 + "\n")
        # --- Main section ---
        f.write(f"  MOLECULAR REPORT: {name}\n")
        f.write("=" * 70 + "\n")

        f.write(f"SMILES: {smiles}\n")
        f.write(f"Molecular formula: {rdMolDescriptors.CalcMolFormula(mol)}\n")
        f.write(f"Number of heavy atoms: {mol.GetNumHeavyAtoms()}\n\n")

        f.write("Morgan algorithm steps:\n")
        for i, (title, ec_dict) in enumerate(morgan.ec_history):
            unique_vals = sorted(set(ec_dict.values()))
            uniq = len(unique_vals)
            f.write(f"  {title}: {uniq} unique values ({', '.join(map(str, unique_vals))})\n")
        f.write("\n")

        # --- Explanation before the table ---
        f.write("Final atom numbering table explanation:\n")
        f.write("  Rank  – final atom numbering assigned by the Morgan algorithm.\n")
        f.write("  Atom  – chemical element symbol of the atom.\n")
        f.write("  Index – internal atom index used by RDKit (starts from 0).\n")
        f.write("           In molecules built from SMILES, indexes follow the order\n")
        f.write("           of atoms as they appear from left to right in the SMILES string.\n")
        f.write("  EC    – final Extended Connectivity value (topological descriptor).\n\n")

        # --- Table ---
        f.write("FINAL ATOM NUMBERING:\n")
        f.write("Rank | Atom | Index | EC\n")
        f.write("-" * 40 + "\n")
        for atom_idx, rank in sorted(final_numbering.items(), key=lambda x: x[1]):
            atom = mol.GetAtomWithIdx(atom_idx)
            f.write(f"{rank:4} | {atom.GetSymbol():4} | {atom_idx:6} | {final_ec[atom_idx]:2}\n")
        
        # --- Date and time the report was created ---
        f.write(f"\nGenerated on: {timestamp}\n\n")

    print(f"Text report created: {path}")

if __name__ == "__main__":
    analyze_molecule()
