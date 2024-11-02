from .config import path_to_class_sz_data
import numpy as np
from .restore_nn import Restore_NN
from .restore_nn import Restore_PCAplusNN
from .suppress_warnings import suppress_warnings
from .emulators_meta_data import *

from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ


cp_tt_nn_jax = {}
cp_te_nn_jax = {}
cp_ee_nn_jax = {}
cp_pp_nn_jax = {}
cp_pknl_nn_jax = {}
cp_pkl_nn_jax = {}
cp_der_nn_jax = {}
cp_da_nn_jax = {}
cp_h_nn_jax = {}
cp_s8_nn_jax = {}


for mp in cosmo_model_list:
    folder, version = split_emulator_string(mp)
    # print(folder, version)
    path_to_emulators = path_to_class_sz_data + '/' + folder +'/'
    
    cp_tt_nn_jax[mp] = Restore_NN(restore_filename=path_to_emulators + 'TTTEEE/' + emulator_dict[mp]['TT'])
    
    cp_te_nn_jax[mp] = Restore_PCAplusNN(restore_filename=path_to_emulators + 'TTTEEE/' + emulator_dict[mp]['TE'])
    
    with suppress_warnings():
        cp_ee_nn_jax[mp] = Restore_NN(restore_filename=path_to_emulators + 'TTTEEE/' + emulator_dict[mp]['EE'])
    
    cp_pp_nn_jax[mp] = Restore_NN(restore_filename=path_to_emulators + 'PP/' + emulator_dict[mp]['PP'])
    
    cp_pknl_nn_jax[mp] = Restore_NN(restore_filename=path_to_emulators + 'PK/' + emulator_dict[mp]['PKNL'])
    
    cp_pkl_nn_jax[mp] = Restore_NN(restore_filename=path_to_emulators + 'PK/' + emulator_dict[mp]['PKL'])
    
    cp_der_nn_jax[mp] = Restore_NN(restore_filename=path_to_emulators + 'derived-parameters/' + emulator_dict[mp]['DER'])
    
    cp_da_nn_jax[mp] = Restore_NN(restore_filename=path_to_emulators + 'growth-and-distances/' + emulator_dict[mp]['DAZ'])
    
    # print(path_to_emulators + 'growth-and-distances/' + emulator_dict[mp]['HZ'])
    emulator_custom = CPJ(probe='custom_log',filepath=path_to_emulators + 'growth-and-distances/' + emulator_dict[mp]['HZ'] + '.npz')
    # print(emulator_custom.parameters)
    # exit()

    cp_h_nn_jax[mp] = CPJ(probe='custom_log',filepath=path_to_emulators + 'growth-and-distances/' + emulator_dict[mp]['HZ'] + '.npz')

    cp_s8_nn_jax[mp] = Restore_NN(restore_filename=path_to_emulators + 'growth-and-distances/' + emulator_dict[mp]['S8Z'])
    
