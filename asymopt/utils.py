from rdkit import Chem
import numpy as np
def clear_ferr_chirality(smi):
    mol = Chem.MolFromSmiles(smi)
    atoms = mol.GetAtoms()
    Fe_atoms = [atom for atom in atoms if atom.GetSymbol() == 'Fe' or atom.GetSymbol() == 'Ru']
    for Fe_at in Fe_atoms:
        nei_atoms = Fe_at.GetNeighbors()
        [atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED) for atom in nei_atoms]
    return Chem.MolToSmiles(mol)

def clear_ada_chirality(smi):
    adamantyl = 'C12CC3CC(C2)CC(C1)C3'
    mol = Chem.MolFromSmiles(smi)
    match_idx_lst = mol.GetSubstructMatches(Chem.MolFromSmiles(adamantyl))
    idx_lst = [0,2,4,7]
    for match_idx in match_idx_lst:
        for at_idx in idx_lst:
            atom = mol.GetAtomWithIdx(match_idx[at_idx])
            atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
    return Chem.MolToSmiles(mol)

def ee2ddG(ee,T):
    '''
    Transformation from ee to ΔΔG
    Parameters
    ----------
    ee : ndarray
        Enantiomeric excess.
    T : ndarray or float
        Temperature (K).
    Returns
    -------
    ddG : ndarray
        ΔΔG (kcal/mol).
    '''
    
    ddG = np.abs(8.314 * T * np.log((1-ee)/(1+ee)))  # J/mol
    ddG = ddG/1000/4.18            # kcal/mol
    return ddG

def ddG2ee(ddG,T):
    '''
    Transformation from ΔΔG to ee. 
    Parameters
    ----------
    ddG : ndarray
        ΔΔG (kcal/mol).
    T : ndarray or float
        Temperature (K).
    Returns
    -------
    ee : ndarray
        Absolute value of enantiomeric excess.
    '''
    
    ddG = ddG*1000*4.18
    ee = (1-np.exp(ddG/(8.314*T)))/(1+np.exp(ddG/(8.314*T)))
    return np.abs(ee)
