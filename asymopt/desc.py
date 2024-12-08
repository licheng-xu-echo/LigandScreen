import os,shutil
from subprocess import run,PIPE
import numpy as np
from rdkit import RDLogger
from rdkit.Chem import rdMolDescriptors,Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from morfeus import BiteAngle, Sterimol, read_xyz,ConeAngle,SASA,BuriedVolume,SolidAngle,VisibleVolume
descs = [desc_name[0] for desc_name in Descriptors._descList]
desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(descs)
RDLogger.DisableLog('rdApp.*')

def maxminscale(array,return_scale=False):
    '''
    Max-min scaler
    Parameters
    ----------
    array : ndarray
        Original numpy array.
    Returns
    -------
    array : ndarray
        numpy array with max-min scaled.
    '''
    if not return_scale:
        return (array - array.min(axis=0))/(array.max(axis=0)-array.min(axis=0))
    else:
        return (array - array.min(axis=0))/(array.max(axis=0)-array.min(axis=0)),array.max(axis=0),array.min(axis=0)

def process_desc(array,return_idx=False):

    array = np.array(array,dtype=np.float32)
    desc_len = array.shape[1]
    rig_idx = []
    for i in range(desc_len):
        try:
            desc_range = array[:,i].max() - array[:,i].min()
            if desc_range != 0 and not np.isnan(desc_range) and not np.isinf(desc_range):
                rig_idx.append(i)
        except:
            continue
    array = array[:,rig_idx]
    if return_idx == False:
        return array
    else:
        return array,rig_idx

def getmorganfp(mol,radius=2,nBits=2048,useChirality=True):
    '''
    
    Parameters
    ----------
    mol : mol
        RDKit mol object.
    Returns
    -------
    mf_desc_map : ndarray
        ndarray of molecular fingerprint descriptors.
    '''
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol,radius=radius,nBits=nBits,useChirality=useChirality)
    return np.array(list(map(eval,list(fp.ToBitString()))))

def getPhysChemDesc(mol):
    return list(desc_calc.CalcDescriptors(mol))

