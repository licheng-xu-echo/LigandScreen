from rdkit import Chem
from rdkit.Chem import AllChem
from .utils import clear_ferr_chirality,clear_ada_chirality

def link_lig_to_metal(lig_mol,metal_type,coord_at_idx_lst):
    rw_lig_mol = Chem.RWMol(lig_mol)
    rw_lig_mol.UpdatePropertyCache(strict=False)
    metal_idx = rw_lig_mol.AddAtom(Chem.Atom(metal_type))
    for at_idx in coord_at_idx_lst:
        rw_lig_mol.AddBond(at_idx,metal_idx,Chem.BondType.DATIVE)
    rw_lig_mol.UpdatePropertyCache(strict=True)
    cat_mol = rw_lig_mol.GetMol()
    return cat_mol

def find_sub_attach_atom(mol, sub_smi):
    """
    Detect whether a specific substituent is present in a molecule, and return the indices of 
    the atoms directly connected to the substituent, as well as the index of the substituent 
    itself. 
    """
    substituent = Chem.MolFromSmiles(sub_smi)
    matches = mol.GetSubstructMatches(substituent)
    if not matches:
        return [],[]

    attach_idx_lst = []
    matches_withH = [] 
    for match in matches:
        H_atom_lst = []
        for atom_idx in match:
            atom = mol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() == 'H':
                    H_atom_lst.append(neighbor.GetIdx())
        match = list(match) + H_atom_lst
        for atom_idx in match:
            atom = mol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                if neighbor.GetIdx() not in match:
                    attach_idx_lst.append(neighbor.GetIdx())
        matches_withH.append(match)
    return attach_idx_lst,matches_withH


def gen_metal_sub_lig_complex(metal_lig_mol,r_1_smi,r_2_smi):
    X_smi_lst = ['OS(=O)(C(F)(F)F)=O','Cl','Br','I']
    B_smi_lst = ['CC1(C)OBOC1(C)C','C1OBOC1','[H]OBO[H]','FB(F)F.[K+]']
    metal_lig_mol = AllChem.AddHs(metal_lig_mol)
    rw_metal_lig_mol = Chem.RWMol(metal_lig_mol)
    metal_idx = [idx for idx,atm in enumerate(metal_lig_mol.GetAtoms()) if atm.GetSymbol() in ['Pd','Ni']][0]
    # process halide part
    rct_1_mol = AllChem.AddHs(Chem.MolFromSmiles(r_1_smi))
    rct_1_at_num = rct_1_mol.GetNumAtoms()
    for X_smi in X_smi_lst:
        attach_idx_lst,group_idx_lst = find_sub_attach_atom(rct_1_mol,X_smi)
        if attach_idx_lst != []:
            break
    if attach_idx_lst == []:
        print(f'[ERROR] Cannot find X group for {r_1_smi}')
    X_attach_idx = attach_idx_lst[0]
    X_group_idx = group_idx_lst[0]

    rw_rct_1_mol = Chem.RWMol(rct_1_mol)
    rw_rct_1_mol.UpdatePropertyCache(strict=False)
    for idx in sorted(X_group_idx,reverse=True):
        if idx < X_attach_idx:
            X_attach_idx -= 1
        rw_rct_1_mol.RemoveAtom(idx)
    # process boronic acid part
    rct_2_mol = AllChem.AddHs(Chem.MolFromSmiles(r_2_smi))
    rct_2_at_num = rct_2_mol.GetNumAtoms()
    for B_smi in B_smi_lst:
        attach_idx_lst,group_idx_lst = find_sub_attach_atom(rct_2_mol,B_smi)
        if attach_idx_lst != []:
            break
    if attach_idx_lst == []:
        print(f'[ERROR] Cannot find B group for {r_2_smi}')
    B_attach_idx = attach_idx_lst[0]
    B_group_idx = group_idx_lst[0]
    rw_rct_2_mol = Chem.RWMol(rct_2_mol)
    rw_rct_2_mol.UpdatePropertyCache(strict=False)
    for idx in sorted(B_group_idx,reverse=True):
        if idx < B_attach_idx:
            B_attach_idx -= 1
        rw_rct_2_mol.RemoveAtom(idx)
    rw_rct_1_mol.InsertMol(rw_rct_2_mol)
    rw_rct_1_mol.InsertMol(rw_metal_lig_mol)
    rw_rct_1_mol.AddBond(X_attach_idx,
                        metal_idx+rct_1_at_num+rct_2_at_num-len(X_group_idx)-len(B_group_idx),
                        Chem.BondType.SINGLE)
    rw_rct_1_mol.AddBond(B_attach_idx+rct_1_at_num-len(X_group_idx),
                        metal_idx+rct_1_at_num+rct_2_at_num-len(X_group_idx)-len(B_group_idx),
                        Chem.BondType.SINGLE)
    metal_lig_sub_comp_mol = rw_rct_1_mol.GetMol()
    return metal_lig_sub_comp_mol


def build_metal_sub_lig_comp_mol(r_1_smi,r_2_smi,metal,lig_smi):
    lig_smi = clear_ferr_chirality(lig_smi)
    lig_smi = clear_ada_chirality(lig_smi)
    lig_mol = Chem.MolFromSmiles(lig_smi)
    atoms = [atom for atom in lig_mol.GetAtoms()]
    coord_atm_idx = []
    if metal == 'Ni':
        for idx,atom in enumerate(atoms):
            if atom.GetSymbol() == 'P':
                coord_atm_idx.append(idx)
            elif atom.GetSymbol() == 'S':
                nei_at_lst = atom.GetNeighbors()
                nei_bond_lst = [lig_mol.GetBondBetweenAtoms(atom.GetIdx(),nei_at.GetIdx()).GetBondType() for nei_at in nei_at_lst]
                if Chem.rdchem.BondType.DOUBLE in nei_bond_lst:
                    coord_atm_idx.append(idx)
        cat_mol = link_lig_to_metal(lig_mol,metal,coord_atm_idx)
        cat_mol = gen_metal_sub_lig_complex(cat_mol,r_1_smi,r_2_smi)
    else:
        # stupid code for now :(
        if lig_mol.HasSubstructMatch(Chem.MolFromSmiles('Oc1c(c2c(P)cccc2OCCO3)c3ccc1')):
            match_idx = lig_mol.GetSubstructMatch(Chem.MolFromSmiles('Oc1c(c2c(P)cccc2OCCO3)c3ccc1'))
            coord_atm_idx = [match_idx[i] for i in [0,5]]
            cat_mol = link_lig_to_metal(lig_mol,metal,coord_atm_idx)
            cat_mol = gen_metal_sub_lig_complex(cat_mol,r_1_smi,r_2_smi)
        elif lig_mol.HasSubstructMatch(Chem.MolFromSmiles('Oc1c(c2c(PCO3)c3ccc2)c(O)ccc1')):
            match_idx = lig_mol.GetSubstructMatch(Chem.MolFromSmiles('Oc1c(c2c(PCO3)c3ccc2)c(O)ccc1'))
            coord_atm_idx = [match_idx[i] for i in [0,5]]
            cat_mol = link_lig_to_metal(lig_mol,metal,coord_atm_idx)
            cat_mol = gen_metal_sub_lig_complex(cat_mol,r_1_smi,r_2_smi)
        elif lig_mol.HasSubstructMatch(Chem.MolFromSmiles('Oc(cc1)c(c(c(P)cc2)c3c2cccc3)c4c1cccc4')):
            match_idx = lig_mol.GetSubstructMatch(Chem.MolFromSmiles('Oc(cc1)c(c(c(P)cc2)c3c2cccc3)c4c1cccc4'))
            coord_atm_idx = [match_idx[i] for i in [0,7]]
            cat_mol = link_lig_to_metal(lig_mol,metal,coord_atm_idx)
            cat_mol = gen_metal_sub_lig_complex(cat_mol,r_1_smi,r_2_smi)
        elif lig_mol.HasSubstructMatch(Chem.MolFromSmiles('C[C@@H]1C[C@H](Oc2c(c3c(N)cccc3O1)c(P)ccc2)C')):
            match_idx = lig_mol.GetSubstructMatch(Chem.MolFromSmiles('C[C@@H]1C[C@H](Oc2c(c3c(N)cccc3O1)c(P)ccc2)C'))
            coord_atm_idx = [match_idx[i] for i in [9,16]]
            cat_mol = link_lig_to_metal(lig_mol,metal,coord_atm_idx)
            cat_mol = gen_metal_sub_lig_complex(cat_mol,r_1_smi,r_2_smi)
        elif lig_mol.HasSubstructMatch(Chem.MolFromSmiles('N1(/N=C/c2ncccc2)CCCC1')):
            match_idx = lig_mol.GetSubstructMatch(Chem.MolFromSmiles('N1(/N=C/c2ncccc2)CCCC1'))
            coord_atm_idx = [match_idx[i] for i in [1,4]]
            cat_mol = link_lig_to_metal(lig_mol,metal,coord_atm_idx)
            cat_mol = gen_metal_sub_lig_complex(cat_mol,r_1_smi,r_2_smi)
        elif lig_mol.HasSubstructMatch(Chem.MolFromSmiles('N1([C@@H]2CCCC[C@H]2N3[C]Nc4c3cccc4)[C]Nc5c1cccc5')):
            match_idx = lig_mol.GetSubstructMatch(Chem.MolFromSmiles('N1([C@@H]2CCCC[C@H]2N3[C]Nc4c3cccc4)[C]Nc5c1cccc5'))
            coord_atm_idx = [match_idx[i] for i in [8,16]]
            cat_mol = link_lig_to_metal(lig_mol,metal,coord_atm_idx)
            cat_mol = gen_metal_sub_lig_complex(cat_mol,r_1_smi,r_2_smi)
        elif lig_mol.HasSubstructMatch(Chem.MolFromSmiles('N[C]NCCCP(c1ccccc1)c2ccccc2')):
            match_idx = lig_mol.GetSubstructMatch(Chem.MolFromSmiles('N[C]NCCCP(c1ccccc1)c2ccccc2'))
            coord_atm_idx = [match_idx[i] for i in [1,6]]
            cat_mol = link_lig_to_metal(lig_mol,metal,coord_atm_idx)
            cat_mol = gen_metal_sub_lig_complex(cat_mol,r_1_smi,r_2_smi)
        elif lig_mol.HasSubstructMatch(Chem.MolFromSmiles('OCC12[C@H]3C4C5C1([Fe]45326789C%10C6C7C8C%109)P')):
            match_idx = lig_mol.GetSubstructMatch(Chem.MolFromSmiles('OCC12[C@H]3C4C5C1([Fe]45326789C%10C6C7C8C%109)P'))
            coord_atm_idx = [match_idx[i] for i in [0,13]]
            cat_mol = link_lig_to_metal(lig_mol,metal,coord_atm_idx)
            cat_mol = gen_metal_sub_lig_complex(cat_mol,r_1_smi,r_2_smi)
        elif lig_mol.HasSubstructMatch(Chem.MolFromSmiles('NCC1([Fe]23456789C%10C6C7C8C%109)[C@@H]4C2C3C15P(c%11ccccc%11)c%12ccccc%12')):
            match_idx = lig_mol.GetSubstructMatch(Chem.MolFromSmiles('NCC1([Fe]23456789C%10C6C7C8C%109)[C@@H]4C2C3C15P(c%11ccccc%11)c%12ccccc%12'))
            coord_atm_idx = [match_idx[i] for i in [0,13]]
            cat_mol = link_lig_to_metal(lig_mol,metal,coord_atm_idx)
            cat_mol = gen_metal_sub_lig_complex(cat_mol,r_1_smi,r_2_smi)
        elif lig_mol.HasSubstructMatch(Chem.MolFromSmiles('c1(Cc2c(N[C]Nc3c(Cc4ccccc4)cccc3Cc5ccccc5)c(Cc6ccccc6)ccc2)ccccc1')):
            match_idx = lig_mol.GetSubstructMatch(Chem.MolFromSmiles('c1(Cc2c(N[C]Nc3c(Cc4ccccc4)cccc3Cc5ccccc5)c(Cc6ccccc6)ccc2)ccccc1'))
            coord_atm_idx = [match_idx[i] for i in [5]]
            cat_mol = link_lig_to_metal(lig_mol,metal,coord_atm_idx)
            cat_mol = gen_metal_sub_lig_complex(cat_mol,r_1_smi,r_2_smi)
        elif lig_mol.HasSubstructMatch(Chem.MolFromSmiles('c1(c2c(c3ccccc3)c(c4ccccc4)c(c5c(p(o6)oc7ccc(cccc8)c8c7c9c6ccc%10c9cccc%10)cccc5)c(c%11ccccc%11)c2c%12ccccc%12)ccccc1')):
            match_idx = lig_mol.GetSubstructMatch(Chem.MolFromSmiles('c1(c2c(c3ccccc3)c(c4ccccc4)c(c5c(p(o6)oc7ccc(cccc8)c8c7c9c6ccc%10c9cccc%10)cccc5)c(c%11ccccc%11)c2c%12ccccc%12)ccccc1'))
            coord_atm_idx = [match_idx[i] for i in [19]]
            cat_mol = link_lig_to_metal(lig_mol,metal,coord_atm_idx)
            cat_mol = gen_metal_sub_lig_complex(cat_mol,r_1_smi,r_2_smi)
        elif lig_mol.HasSubstructMatch(Chem.MolFromSmiles('COC(CC[C@H]([C@H]1CCC2C3CCC4C[C@@H](CC[C@@]4(C3C[C@@H]([C@@]21C)Op(oc5c6cccc5)oc7c6cccc7)C)OC(C)=O)C)=O')):
            match_idx = lig_mol.GetSubstructMatch(Chem.MolFromSmiles('COC(CC[C@H]([C@H]1CCC2C3CCC4C[C@@H](CC[C@@]4(C3C[C@@H]([C@@]21C)Op(oc5c6cccc5)oc7c6cccc7)C)OC(C)=O)C)=O'))
            coord_atm_idx = [match_idx[i] for i in [25]]
            cat_mol = link_lig_to_metal(lig_mol,metal,coord_atm_idx)
            cat_mol = gen_metal_sub_lig_complex(cat_mol,r_1_smi,r_2_smi)
        elif lig_mol.HasSubstructMatch(Chem.MolFromSmiles('c1(C2=NCCO2)ncccc1')):
            match_idx = lig_mol.GetSubstructMatch(Chem.MolFromSmiles('c1(C2=NCCO2)ncccc1'))
            coord_atm_idx = [match_idx[i] for i in [2,6]]
            cat_mol = link_lig_to_metal(lig_mol,metal,coord_atm_idx)
            cat_mol = gen_metal_sub_lig_complex(cat_mol,r_1_smi,r_2_smi)
        elif lig_mol.HasSubstructMatch(Chem.MolFromSmiles('COc1c(C23[C@@H]4C5C6C2([Fe]5643789%10C%11C7C8C9C%11%10)P(C%12CCCCC%12)C%13CCCCC%13)cccc1')):
            match_idx = lig_mol.GetSubstructMatch(Chem.MolFromSmiles('COc1c(C23[C@@H]4C5C6C2([Fe]5643789%10C%11C7C8C9C%11%10)P(C%12CCCCC%12)C%13CCCCC%13)cccc1'))
            coord_atm_idx = [match_idx[i] for i in [15]]
            cat_mol = link_lig_to_metal(lig_mol,metal,coord_atm_idx)
            cat_mol = gen_metal_sub_lig_complex(cat_mol,r_1_smi,r_2_smi)
        elif lig_mol.HasSubstructMatch(Chem.MolFromSmiles('CC(C)(OC1CO2)OC1COP2N(Cc3ccccc3)Cc4ccccc4')):
            match_idx = lig_mol.GetSubstructMatch(Chem.MolFromSmiles('CC(C)(OC1CO2)OC1COP2N(Cc3ccccc3)Cc4ccccc4'))
            coord_atm_idx = [match_idx[i] for i in [11]]
            cat_mol = link_lig_to_metal(lig_mol,metal,coord_atm_idx)
            cat_mol = gen_metal_sub_lig_complex(cat_mol,r_1_smi,r_2_smi)
        elif lig_mol.HasSubstructMatch(Chem.MolFromSmiles('Pc1c(C(N)=O)cccc1')):
            match_idx = lig_mol.GetSubstructMatch(Chem.MolFromSmiles('Pc1c(C(N)=O)cccc1'))
            coord_atm_idx = [match_idx[i] for i in [0]]
            cat_mol = link_lig_to_metal(lig_mol,metal,coord_atm_idx)
            cat_mol = gen_metal_sub_lig_complex(cat_mol,r_1_smi,r_2_smi)
        elif lig_mol.HasSubstructMatch(Chem.MolFromSmiles('c1(P(C23C4C5C6C2[Fe]5643789%10C%11C7C8(C9[C@H]%11%10)P(c%12ccccc%12)c%13ccccc%13)c%14ccccc%14)ccccc1')):
            match_idx = lig_mol.GetSubstructMatch(Chem.MolFromSmiles('c1(P(C23C4C5C6C2[Fe]5643789%10C%11C7C8(C9[C@H]%11%10)P(c%12ccccc%12)c%13ccccc%13)c%14ccccc%14)ccccc1'))
            coord_atm_idx = [match_idx[i] for i in [1,13]]
            cat_mol = link_lig_to_metal(lig_mol,metal,coord_atm_idx)
            cat_mol = gen_metal_sub_lig_complex(cat_mol,r_1_smi,r_2_smi)
        elif lig_mol.HasSubstructMatch(Chem.MolFromSmiles('C12=CC=CC=C1C=CC(P(C3=CC=CC=C3)C4=CC=CC=C4)=[C@@]2[C@@]5=C(C=CC=C6)C6=CC=C5P(C7=CC=CC=C7)C8=CC=CC=C8')):
            match_idx = lig_mol.GetSubstructMatch(Chem.MolFromSmiles('C12=CC=CC=C1C=CC(P(C3=CC=CC=C3)C4=CC=CC=C4)=[C@@]2[C@@]5=C(C=CC=C6)C6=CC=C5P(C7=CC=CC=C7)C8=CC=CC=C8'))
            coord_atm_idx = [match_idx[i] for i in [9,33]]
            cat_mol = link_lig_to_metal(lig_mol,metal,coord_atm_idx)
            cat_mol = gen_metal_sub_lig_complex(cat_mol,r_1_smi,r_2_smi)     
        elif lig_mol.HasSubstructMatch(Chem.MolFromSmiles('NC1=[C@@]([C@@]2=C(C=CC=C3)C3=CC=C2N)C4=CC=CC=C4C=C1')):
            match_idx = lig_mol.GetSubstructMatch(Chem.MolFromSmiles('NC1=[C@@]([C@@]2=C(C=CC=C3)C3=CC=C2N)C4=CC=CC=C4C=C1'))
            coord_atm_idx = [match_idx[i] for i in [0,13]]
            cat_mol = link_lig_to_metal(lig_mol,metal,coord_atm_idx)
            cat_mol = gen_metal_sub_lig_complex(cat_mol,r_1_smi,r_2_smi)
        elif lig_mol.HasSubstructMatch(Chem.MolFromSmiles('PC1=[C@]([C@]2=C(C=CC=C3)C3=CC=C2N)C4=CC=CC=C4C=C1')):
            match_idx = lig_mol.GetSubstructMatch(Chem.MolFromSmiles('PC1=[C@]([C@]2=C(C=CC=C3)C3=CC=C2N)C4=CC=CC=C4C=C1'))
            coord_atm_idx = [match_idx[i] for i in [0,13]]
            cat_mol = link_lig_to_metal(lig_mol,metal,coord_atm_idx)
            cat_mol = gen_metal_sub_lig_complex(cat_mol,r_1_smi,r_2_smi)
    return cat_mol