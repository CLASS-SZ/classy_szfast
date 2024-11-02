from .config import path_to_class_sz_data
import numpy as np
from .restore_nn import Restore_NN
from .restore_nn import Restore_PCAplusNN
from .suppress_warnings import suppress_warnings
from .emulators_meta_data import *


cp_tt_nn = {}
cp_te_nn = {}
cp_ee_nn = {}
cp_pp_nn = {}
cp_pknl_nn = {}
cp_pkl_nn = {}
cp_pkl_fftlog_alphas_real_nn = {}
cp_pkl_fftlog_alphas_imag_nn = {}
cp_pkl_fftlog_alphas_nus = {}
cp_der_nn = {}
cp_da_nn = {}
cp_h_nn = {}
cp_s8_nn = {}


for mp in cosmo_model_list:
    folder, version = split_emulator_string(mp)
    # print(folder, version)
    path_to_emulators = path_to_class_sz_data + '/' + folder +'/'
    
    cp_tt_nn[mp] = Restore_NN(restore_filename=path_to_emulators + 'TTTEEE/' + emulator_dict[mp]['TT'])
    
    cp_te_nn[mp] = Restore_PCAplusNN(restore_filename=path_to_emulators + 'TTTEEE/' + emulator_dict[mp]['TE'])
    
    with suppress_warnings():
        cp_ee_nn[mp] = Restore_NN(restore_filename=path_to_emulators + 'TTTEEE/' + emulator_dict[mp]['EE'])
    
    cp_pp_nn[mp] = Restore_NN(restore_filename=path_to_emulators + 'PP/' + emulator_dict[mp]['PP'])
    
    cp_pknl_nn[mp] = Restore_NN(restore_filename=path_to_emulators + 'PK/' + emulator_dict[mp]['PKNL'])
    
    cp_pkl_nn[mp] = Restore_NN(restore_filename=path_to_emulators + 'PK/' + emulator_dict[mp]['PKL'])

    if (mp == 'lcdm') and (dofftlog_alphas == True):
        cp_pkl_fftlog_alphas_real_nn[mp] = Restore_PCAplusNN(restore_filename=path_to_emulators + 'PK/' + emulator_dict[mp]['PKLFFTLOG_ALPHAS_REAL']
                                 )
        cp_pkl_fftlog_alphas_imag_nn[mp] = Restore_PCAplusNN(restore_filename=path_to_emulators + 'PK/' + emulator_dict[mp]['PKLFFTLOG_ALPHAS_IMAG']
                                 )
        cp_pkl_fftlog_alphas_nus[mp] = np.load(path_to_emulators + 'PK/PKL_FFTLog_alphas_nu_v1.npz')
    
    cp_der_nn[mp] = Restore_NN(restore_filename=path_to_emulators + 'derived-parameters/' + emulator_dict[mp]['DER'])
    
    cp_da_nn[mp] = Restore_NN(restore_filename=path_to_emulators + 'growth-and-distances/' + emulator_dict[mp]['DAZ'])
    
    cp_h_nn[mp] = Restore_NN(restore_filename=path_to_emulators + 'growth-and-distances/' + emulator_dict[mp]['HZ'])
    
    cp_s8_nn[mp] = Restore_NN(restore_filename=path_to_emulators + 'growth-and-distances/' + emulator_dict[mp]['S8Z'])
    



