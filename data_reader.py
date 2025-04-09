import os
from rdkit import Chem
import selfies as sf
import numpy as np
import pickle as pkl
path = "../../../../forth_year/project/data/datasets/qm9bg"


#
#This function takes a list of shifts and divides this spectrum into a series of boolean buckets
#

def spectrum_binning(peaks, nuclei):
    #print(f"starting with peaks = {peaks},  and nuclei = {nuclei}")
    #sets the number of buckets in the spectrum
    num_buckets = 1000

    #sets the shift ranges depending on whether we're dealing with proton or carbon
    if nuclei == 1:
        spectrum_range = np.array([-2,12])
    if nuclei == 12:
        spectrum_range = np.array([-20, 240])
    
    #sets up bucketing
    buckets = np.zeros(num_buckets, dtype=int) 
    bucket_size = (spectrum_range[1] - spectrum_range[0]) / num_buckets
    
    for peak in peaks:
        bucket_index = int((peak - spectrum_range[0]) / bucket_size)
        if 0 <= bucket_index < num_buckets:  # Ensure index is within bounds
            buckets[bucket_index] = 1
    
    return buckets

#
#this function reads an sdf file and gets scaled proton and carbon spectra out of it
#returns a dictionary in the form {"nucleus": [shift1, shift2, ...], ...}
#

def sdf_to_spectrum(mol):

    #adds the hydrogens to the molecule
    mol = Chem.AddHs(mol)

    #loops through the atoms in the molecule and hunts for hydrogens
    hydrogen_list = []
    
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            atom_number = atom.GetIdx()
            hydrogen_list.append(atom_number)


    carbon_list = []
    
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6:
            atom_number = atom.GetIdx()
            carbon_list.append(atom_number)
    
    #creates a dictionary which uses the SDF arbritary atom number as a key and the atom type as the data
    atom_type_dict = {}
    for atom in mol.GetAtoms():
        atom_type_dict[atom.GetIdx()] =  atom.GetAtomicNum()

    
    import re

    #gets a string which contains all the data stored past NMREDATA_ASSIGNMENT
    raw_data = str(mol.GetProp("NMREDATA_ASSIGNMENT"))

    # Extract NMR shifts
    nmr_shifts = re.findall(r'(\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*\d+\s*,\s*-?\d+\.\d+', raw_data)

    # Convert the extracted data into a dictionary
    nmr_shift_dict = {int(match[0]): float(match[1]) for match in nmr_shifts}
    #print(f"printing NMR SHIFT DICT {nmr_shift_dict}")
    ############################################################################
    #                                                                          #
    #          Correcting the shifts from DFT to literature emulated           #
    #                                                                          #
    ############################################################################

    #                     A        B
    #pymol scaling is [-1.0594, 32.2293]
    # true shift = (DFT_shift - B / A)

    #carbon scaling
    #6: [-1.0207, 187.4436]

    #a function which converts from dft shift to experimental
    def dft_scaler(dft_shift, atom_type=1 ):
        #stores and assigns the shift scaling constants for Hydrogen and Carbon atoms
        if atom_type == 1:
            A = -1.0594
            B = 32.2293
        if atom_type == 6:
            A = -1.0207
            B = 187.4436
        corrected_shift = (dft_shift - B) / A
        return corrected_shift


    #loops through the H1 shifts and interpolates each value to one which matches experimental data
    corrected_shift_list = []

    #hacky error handling to stop molecules crashing that have atoms indeces in their atom index list which are out of range for the nmr_shift_dict
    for atom_index in hydrogen_list:
        try:
            nmr_shift_dict[atom_index] = dft_scaler(nmr_shift_dict[atom_index])
            corrected_shift_list.append(nmr_shift_dict[atom_index])
        except:

            return "failed"
    
    carbon_corrected_shift_list = []

    #loops througb the list of carbon atoms and scales the dft 
    for atom_index in carbon_list:
        try:
            nmr_shift_dict[atom_index] = dft_scaler(nmr_shift_dict[atom_index], atom_type=6)
            carbon_corrected_shift_list.append(nmr_shift_dict[atom_index])
            
        except:
            print(f"ERROR molecule with index x failed to compute because of index ranging issues")
            return "failed"
    # print(f"Corrected Carbon list : {carbon_corrected_shift_list}")
    # print(f"corrected shift list: {corrected_shift_list}")

    return {"proton": corrected_shift_list, "carbon": carbon_corrected_shift_list}



molecule_data = {}

for batch in os.listdir(path)[:-1]:

    for sdf_file_name in os.listdir(f"{path}/{batch}"):
        qm9_index = sdf_file_name[6:-13]

        single_mol_data = {}

    
        sdf_file_path = f"{path}/{batch}/{sdf_file_name}"

        try:
            with Chem.SDMolSupplier(sdf_file_path) as sdf_supplier:
                # Iterate over molecules in the SDF file
                for mol in sdf_supplier:
                    if mol is not None:
                        # Do something with the molecule (e.g., print or process it)
                        mol.GetPropsAsDict()

                ##use an rdkit atribute to get smile uistrings
                smile_string = Chem.MolToSmiles(mol)
                single_mol_data["smile"] = smile_string
                
                #convert smile to selfie
                selfie = sf.encoder(smile_string)
                single_mol_data["selfie"] = selfie



                #gets the shifts
                #this returns a dictionary of the form {"nucleus": [shift1, shift2, ...], ...}

                shifts = sdf_to_spectrum(mol)
                single_mol_data["shifts"] = shifts

                #gets binned proton spectrum
                binned_proton = spectrum_binning(shifts["proton"], 1)
                single_mol_data["binned_proton"] = binned_proton
                
                #gets binned carbon spectrum
                binned_carbon = carbon = spectrum_binning(shifts["carbon"], 12)
                single_mol_data["binned_carbon"] = binned_carbon
            

                

                #adds the mmolecules data to the molecule dictionary
                molecule_data[str(qm9_index)] = single_mol_data
        except:
            print("molecule could not be converted to smile")



#print(molecule_data)

# Save to a .pkl file
print("processing complete!")
with open("data/spectra.pkl", "wb") as f:
    pkl.dump(molecule_data, f)

print("Save Complete")


