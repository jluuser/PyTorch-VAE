import numpy as np
import os
import argparse
from Bio.PDB import PDBIO, Structure, Model, Chain, Atom, Residue
'''
python npy_to_pdb.py --input_dir /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/results/aeot_runs/test_run_random_02/filtered_npy --output_dir /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/results/aeot_runs/test_run_random_02/filtered_pdb
'''    
def write_pdb(coords: np.array, pdb_file):
    structure = Structure.Structure("example")
    model = Model.Model(0)
    structure.add(model)
    chain = Chain.Chain("A")
    model.add(chain)
    for i, coord in enumerate(coords):
        residue = Residue.Residue((" ", i + 1, " "), "GLY", "")
        atom = Atom.Atom("CA", coord, 1.0, 1.0, " ", "CA", i + 1, "C")
        residue.add(atom)
        chain.add(residue)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(pdb_file))

def convert_npy_to_pdb(input_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
    print("Found " + str(len(files)) + " .npy files in " + str(input_dir))
    
    count = 0
    for f in files:
        file_path = os.path.join(input_dir, f)
        
        # Generate output filename: remove '_curve' and change extension to .pdb
        base_name = f.replace("_curve.npy", "").replace(".npy", "")
        output_pdb_path = os.path.join(output_dir, base_name + ".pdb")
        
        try:
            # Load npy file
            data = np.load(file_path, allow_pickle=True)
            
            # If loaded as dictionary (0-d array containing dict)
            if data.ndim == 0:
                data = data.item()
            
            # Check if it's a dict and has curve_coords
            if isinstance(data, dict):
                if 'curve_coords' in data:
                    ca_coords = data['curve_coords']
                elif 'ca_coords' in data:
                    ca_coords = data['ca_coords']
                else:
                    print("Skipping " + f + ": Dictionary missing curve_coords or ca_coords")
                    continue
            # If it's a numpy array
            elif isinstance(data, np.ndarray):
                if data.ndim == 2 and data.shape[1] >= 3:
                    ca_coords = data[:, :3]
                else:
                    print("Skipping " + f + ": Array shape " + str(data.shape) + " not compatible")
                    continue
            else:
                print("Skipping " + f + ": Unknown data structure " + str(type(data)))
                continue

            # Ensure coords is numpy array and check shape again
            ca_coords = np.array(ca_coords)
            if ca_coords.ndim == 2 and ca_coords.shape[1] >= 3:
                # Write to PDB
                write_pdb(ca_coords, output_pdb_path)
                count += 1
                print("Converted: " + f + " -> " + output_pdb_path)
            else:
                print("Skipping " + f + ": Extracted coords shape " + str(ca_coords.shape) + " invalid")
                
        except Exception as e:
            print("Error converting " + f + ": " + str(e))
            
    print("Successfully converted " + str(count) + " files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert npy files to pdb.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing .npy files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save converted pdb files')
    
    args = parser.parse_args()
    
    if os.path.exists(args.input_dir):
        convert_npy_to_pdb(args.input_dir, args.output_dir)
    else:
        print("Directory not found: " + str(args.input_dir))