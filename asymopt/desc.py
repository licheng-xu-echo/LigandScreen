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


class xTBCalculator(object):
    def __init__(self,xtb_dir,xyz_name,chrg=0,gfn=2):
        self.xtb_dir = xtb_dir
        self.xyz_name = xyz_name if '.xyz' in xyz_name else f'{xyz_name}.xyz'
        self.chrg = chrg
        self.gfn = gfn
    def set_xyzfile(self,xyz_file,chrg=0,gfn=2):
        self.xyz_file = xyz_file
        self.chrg = chrg
        self.gfn = gfn
        self.xyz_name = os.path.basename(self.xyz_file)
        self.xyz_dir = os.path.dirname(self.xyz_file)
        self.xtb_dir = f'{self.xyz_dir}/xtb'
        if not os.path.exists(self.xtb_dir):
            os.mkdir(self.xtb_dir)
        if not os.path.exists(f'{self.xtb_dir}/{self.xyz_name[:-4]}'):
            os.mkdir(f'{self.xtb_dir}/{self.xyz_name[:-4]}')
        os.chdir(f'{self.xtb_dir}/{self.xyz_name[:-4]}')
    def optimize(self):
        shutil.copyfile(self.xyz_file,f'{self.xtb_dir}/{self.xyz_name[:-4]}/{self.xyz_name}')
        cmd = f'xtb {self.xyz_name} --opt --gfn {self.gfn} --charge {self.chrg} > xtboptlog'
        if not os.path.exists(f'{self.xtb_dir}/{self.xyz_name[:-4]}/xtbopt.xyz'):
            print(f'[INFO] Optimize {self.xyz_name}')
            ret = run(cmd,shell=True,stdout=PIPE,stderr=PIPE,universal_newlines=True)
    def single_point(self):
        os.chdir(f'{self.xtb_dir}/{self.xyz_name[:-4]}')
        xtbopt_file = f'{self.xtb_dir}/{self.xyz_name[:-4]}/xtbopt.xyz'
        if not os.path.exists(xtbopt_file):
            print(f'[ERROR] xTB optimization task failed ({self.xyz_name})')
            return False
        cmd_lst = [f'xtb xtbopt.xyz --gfn {self.gfn} --charge {self.chrg} --vfukui > vfukuilog',
                   f'xtb xtbopt.xyz --gfn {self.gfn} --charge {self.chrg} --vipea > vipealog',
                   f'xtb xtbopt.xyz --gfn {self.gfn} --charge {self.chrg} --vomega > vomegalog',
                   f'xtb xtbopt.xyz --gfn {self.gfn} --charge {self.chrg} > splog']
        print(f'[INFO] Single point calculation for {self.xyz_name}')
        for cmd in cmd_lst:
            run(cmd,shell=True,stdout=PIPE,stderr=PIPE,universal_newlines=True)
        return True
class xTBDescriptor(object):
    def __init__(self):
        pass
    def read_desc(self,xtb_dir,force=False):
        self.xtb_dir = xtb_dir
        self.charges_f = f'{xtb_dir}/charges'
        self.splog = f'{xtb_dir}/splog'
        self.vfukuilog = f'{xtb_dir}/vfukuilog'
        self.vipealog = f'{xtb_dir}/vipealog'
        self.vomegalog = f'{xtb_dir}/vomegalog'
        self.wbo_f = f'{xtb_dir}/wbo'
        if not os.path.exists(f'{xtb_dir}/xtbopt.xyz'):
            return 
        with open(f'{xtb_dir}/xtbopt.xyz','r') as fr:
            lines = fr.readlines()
        self.atom_num = int(lines[0].strip())
        self.desc_ens = {}
        self.desc_ens['global'] = {}
        self.desc_ens['local'] = {}
        sp_all_done_flag = True
        if force or not self.load():
            if self.is_exists(self.charges_f):
                self.read_charges()
            else:
                sp_all_done_flag = False
            if self.is_exists(self.splog):
                self.read_splog()
            else:
                sp_all_done_flag = False
            if self.is_exists(self.vfukuilog):
                self.read_vfukuilog()
            else:
                sp_all_done_flag = False
            if self.is_exists(self.vipealog):
                self.read_vipealog()
            else:
                sp_all_done_flag = False
            if self.is_exists(self.vomegalog):
                self.read_vomegalog()
            else:
                sp_all_done_flag = False
            if self.is_exists(self.wbo_f):
                self.read_wbo()
            else:
                sp_all_done_flag = False
        if sp_all_done_flag:
            self.save()
    def is_exists(self,file):
        return os.path.exists(file)
    def read_charges(self):
        with open(self.charges_f,'r') as fr:
            lines = fr.readlines()
        charges = np.array([float(chrg.strip()) for chrg in lines])
        self.desc_ens['charges'] = charges
        self.desc_ens['local']['charges'] = charges.reshape(-1,1)
    def read_splog(self):
        with open(self.splog,'r') as fr:
            lines = fr.readlines()
        for i,line in enumerate(lines):
            if '(HOMO)' in line:
                homo = float(line.strip().split()[-2])
                self.desc_ens['HOMO'] = homo
                self.desc_ens['global']['HOMO'] = homo
            elif '(LUMO)' in line:
                lumo = float(line.strip().split()[-2])
                self.desc_ens['LUMO'] = lumo
                self.desc_ens['global']['LUMO'] = lumo
            elif 'TOTAL ENERGY' in line:
                tot_energy = float(line.strip().split()[-3])
                self.desc_ens['E'] = round(tot_energy,8)
                self.desc_ens['global']['E'] = round(tot_energy,8)
            elif 'molecular dipole' in line:
                qx,qy,qz = [float(q) for q in lines[i+2].strip().split()[-3:]]
                full_x,full_y,full_z,full_tot = [float(q) for q in lines[i+3].strip().split()[-4:]]
                self.desc_ens['dipole_q_xyz'] = np.array([qx,qy,qz])
                self.desc_ens['dipole_full_xyz'] = np.array([full_x,full_y,full_z])
                self.desc_ens['dipole_tot'] = full_tot
                self.desc_ens['global']['dipole'] = np.concatenate([np.array([qx,qy,qz]),
                                                                    np.array([full_x,full_y,full_z]),
                                                                    np.array([full_tot])])
            elif '#   Z          covCN         q      C6AA      α(0)' in line:
                q_local_inf = lines[i+1:i+1+self.atom_num]
                convCN = [float(item.strip().split()[3]) for item in q_local_inf]
                q = [float(item.strip().split()[4]) for item in q_local_inf]
                C6AA = [float(item.strip().split()[5]) for item in q_local_inf]
                alpha = [float(item.strip().split()[6]) for item in q_local_inf]
                self.desc_ens['convCN'] = np.array(convCN)
                self.desc_ens['q'] = np.array(q)
                self.desc_ens['C6AA'] = np.array(C6AA)
                self.desc_ens['alpha'] = np.array(alpha)
                
                self.desc_ens['local']['q_related'] = np.concatenate([np.array(convCN).reshape(-1,1),
                                                                    np.array(q).reshape(-1,1),
                                                                    np.array(C6AA).reshape(-1,1),
                                                                    np.array(alpha).reshape(-1,1)],axis=1)
        self.desc_ens['GAP'] = round(self.desc_ens['LUMO'] - self.desc_ens['HOMO'],8)
        self.desc_ens['global']['GAP'] = self.desc_ens['GAP']  
    def read_vfukuilog(self):
        with open(self.vfukuilog,'r') as fr:
            lines = fr.readlines()
        for idx in range(len(lines)):
            if '#        f(+)     f(-)     f(0)' in lines[idx]:
                fukui_inf = lines[idx+1:idx+1+self.atom_num]
                f_p = np.array([float(item.strip().split()[1]) for item in fukui_inf]).reshape(-1,1)
                f_m = np.array([float(item.strip().split()[2]) for item in fukui_inf]).reshape(-1,1)
                f_0 = np.array([float(item.strip().split()[3]) for item in fukui_inf]).reshape(-1,1)
                self.desc_ens['fukui'] = np.concatenate([f_p,f_m,f_0],axis=1)
                self.desc_ens['local']['fukui'] = np.concatenate([f_p,f_m,f_0],axis=1)
    def read_vipealog(self):
        with open(self.vipealog,'r') as fr:
            lines = fr.readlines()
        for line in lines:
            if 'delta SCC IP (eV):' in line:
                vip = float(line.strip().split()[-1]) # vertical IP
                self.desc_ens['VIP'] = vip
                self.desc_ens['global']['VIP'] = vip
            elif 'delta SCC EA (eV):' in line:
                vea = float(line.strip().split()[-1]) # vertical EA
                self.desc_ens['VEA'] = vea
                self.desc_ens['global']['VEA'] = vea
    def read_vomegalog(self):
        with open(self.vomegalog,'r') as fr:
            lines = fr.readlines()
        for line in lines:
            if "Global electrophilicity index (eV):" in line:
                GEI = float(line.strip().split()[-1])
                self.desc_ens['GEI'] = GEI
                self.desc_ens['global']['GEI'] = GEI
    def read_wbo(self):
        with open(self.wbo_f,'r') as fr:
            lines = fr.readlines()
        wbo_inf = {}
        for line in lines:
            atom_i,atom_j,wbo = list(map(eval,line.strip().split()))
            wbo_inf[(atom_i-1,atom_j-1)] = wbo # 0-index
        self.desc_ens['wbo'] = wbo_inf
        self.desc_ens['local']['wbo'] = wbo_inf
    def save(self,path=None):
        if path == None:
            path = f'{self.xtb_dir}/desc_ens.npy'
        np.save(path,self.desc_ens)
    def load(self,path=None):
        if path == None:
            path = f'{self.xtb_dir}/desc_ens.npy'
        try:
            self.desc_ens = np.load(path,allow_pickle=True).item()
            return True
        except:
            return False
        
class StericDescriptor(object):
    '''
    0-index
    '''
    def __init__(self,xyz_file):
        self.xyz_file = xyz_file
        self.elements, self.coordinates = read_xyz(self.xyz_file)
    def BV(self,metal_index, excluded_atoms=None, radii=None, include_hs=False, radius=3.5, radii_type='bondi',
           radii_scale=1.17, density=0.001, z_axis_atoms=None, xz_plane_atoms=None):
        '''
        Buried Volume
        '''
        if excluded_atoms != None:
            excluded_atoms = [idx+1 for idx in excluded_atoms]
        if z_axis_atoms != None:
            z_axis_atoms = [idx+1 for idx in z_axis_atoms]
        if xz_plane_atoms != None:
            xz_plane_atoms = [idx+1 for idx in xz_plane_atoms]
        bv = BuriedVolume(self.elements,self.coordinates,metal_index+1,
                          excluded_atoms, radii, include_hs, radius, radii_type,
                          radii_scale, density, z_axis_atoms, xz_plane_atoms)
        return [bv.fraction_buried_volume]
    def CA(self,atom_1, radii=None, radii_type='crc', method='libconeangle'):
        '''
        Cone Angle
        '''
        ca = ConeAngle(self.elements,self.coordinates,atom_1+1, 
                       radii, radii_type, method)
        return [ca.cone_angle]
    def SASA(self, radii=None, radii_type='crc', probe_radius=1.4, density=0.01):
        sasa = SASA(self.elements,self.coordinates,radii, radii_type, probe_radius, density)
        return [sasa.area]
    def SA(self,metal_index, radii=None, radii_type='crc', density=0.001):
        '''
        Solid Angle
        '''
        sa = SolidAngle(self.elements,self.coordinates,metal_index+1, radii, radii_type, density)
        return [sa.solid_angle]
    def VV(self,metal_index, include_hs=True, radii=None, radii_type='pyykko', radius=3.5, density=0.01):
        '''
        Visible Volume
        '''
        vv = VisibleVolume(self.elements,self.coordinates,metal_index+1, include_hs, 
                           radii, radii_type, radius, density)
        return [vv.distal_visible_area, vv.distal_visible_volume, 
                vv.proximal_visible_area, vv.proximal_visible_volume]
    def BA(self,metal_index, ligand_index_1, ligand_index_2, ref_atoms=None, ref_vector=None):
        '''
        Bite Angle
        '''
        if ref_atoms != None:
            ref_atoms = [idx+1 for idx in ref_atoms]
        if ref_vector != None:
            print('[WARN] Reference Vector is not handled for Bite Angle')
        ba = BiteAngle(self.coordinates,metal_index+1, 
                       ligand_index_1+1, ligand_index_2+1, ref_atoms, ref_vector)
        return [ba.angle]
    def Sterimol(self,dummy_index,attached_index,radii=None, radii_type='crc', n_rot_vectors=3600, excluded_atoms=None, calculate=True):
        dummy_index = dummy_index + 1
        attached_index = attached_index + 1
        if excluded_atoms != None:
            excluded_atoms = [idx+1 for idx in excluded_atoms]
        sterimol = Sterimol(self.elements, self.coordinates, dummy_index, attached_index, radii=None, radii_type='crc', n_rot_vectors=3600, excluded_atoms=None, calculate=True)
        return [sterimol.B_5_value,sterimol.L_value]
    
