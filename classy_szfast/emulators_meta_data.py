from .suppress_warnings import suppress_warnings
import warnings
from contextlib import contextmanager
import logging

# Suppress absl warnings
import absl.logging
absl.logging.set_verbosity('error')
# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
with suppress_warnings():
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


import re

dofftlog_alphas = False

cosmopower_derived_params_names = ['100*theta_s',
                                   'sigma8',
                                   'YHe',
                                   'z_reio',
                                   'Neff',
                                   'tau_rec',
                                   'z_rec',
                                   'rs_rec',
                                   'ra_rec',
                                   'tau_star',
                                   'z_star',
                                   'rs_star',
                                   'ra_star',
                                   'rs_drag']

cp_l_max_scalars = 11000 # max multipole of training data

cosmo_model_list = [
    'lcdm',
    'mnu',
    'neff',
    'wcdm',
    'ede',
    'mnu-3states',
    'ede-v2'
]

emulator_dict = {}
emulator_dict['lcdm'] = {}
emulator_dict['mnu'] = {}
emulator_dict['neff'] = {}
emulator_dict['wcdm'] = {}
emulator_dict['ede'] = {}
emulator_dict['mnu-3states'] = {}
emulator_dict['ede-v2'] = {}

### note on ncdm:
# N_ncdm : 3
# m_ncdm : 0.02, 0.02, 0.02
# deg_ncdm: 1
# and
# N_ncdm: 1
# deg_ncdm: 3
# m_ncdm : 0.02
# are equivalent but deg_ncdm: 3 is much faster. 


emulator_dict['lcdm']['TT'] = 'TT_v1'
emulator_dict['lcdm']['TE'] = 'TE_v1'
emulator_dict['lcdm']['EE'] = 'EE_v1'
emulator_dict['lcdm']['PP'] = 'PP_v1'
emulator_dict['lcdm']['PKNL'] = 'PKNL_v1'
emulator_dict['lcdm']['PKL'] = 'PKL_v1'
emulator_dict['lcdm']['PKLFFTLOG_ALPHAS_REAL'] = 'PKLFFTLOGALPHAS_creal_v1'
emulator_dict['lcdm']['PKLFFTLOG_ALPHAS_IMAG'] = 'PKLFFTLOGALPHAS_cimag_v1'
emulator_dict['lcdm']['DER'] = 'DER_v1'
emulator_dict['lcdm']['DAZ'] = 'DAZ_v1'
emulator_dict['lcdm']['HZ'] = 'HZ_v1'
emulator_dict['lcdm']['S8Z'] = 'S8Z_v1'
emulator_dict['lcdm']['default'] = {}
emulator_dict['lcdm']['default']['tau_reio'] = 0.054
emulator_dict['lcdm']['default']['H0'] = 67.66
emulator_dict['lcdm']['default']['ln10^{10}A_s'] = 3.047
emulator_dict['lcdm']['default']['omega_b'] = 0.02242
emulator_dict['lcdm']['default']['omega_cdm'] = 0.11933
emulator_dict['lcdm']['default']['n_s'] = 0.9665
emulator_dict['lcdm']['default']['N_ur'] = 2.0328
emulator_dict['lcdm']['default']['N_ncdm'] = 1
emulator_dict['lcdm']['default']['m_ncdm'] = 0.06
emulator_dict['lcdm']['default']['deg_ncdm'] = 1
emulator_dict['lcdm']['default']['T_cmb'] = 2.7255

emulator_dict['mnu']['TT'] = 'TT_mnu_v1'
emulator_dict['mnu']['TE'] = 'TE_mnu_v1'
emulator_dict['mnu']['EE'] = 'EE_mnu_v1'
emulator_dict['mnu']['PP'] = 'PP_mnu_v1'
emulator_dict['mnu']['PKNL'] = 'PKNL_mnu_v1'
emulator_dict['mnu']['PKL'] = 'PKL_mnu_v1'
emulator_dict['mnu']['DER'] = 'DER_mnu_v1'
emulator_dict['mnu']['DAZ'] = 'DAZ_mnu_v1'
emulator_dict['mnu']['HZ'] = 'HZ_mnu_v1'
emulator_dict['mnu']['S8Z'] = 'S8Z_mnu_v1'
emulator_dict['mnu']['default'] = {}
emulator_dict['mnu']['default']['tau_reio'] = 0.054
emulator_dict['mnu']['default']['H0'] = 67.66
emulator_dict['mnu']['default']['ln10^{10}A_s'] = 3.047
emulator_dict['mnu']['default']['omega_b'] = 0.02242
emulator_dict['mnu']['default']['omega_cdm'] = 0.11933
emulator_dict['mnu']['default']['n_s'] = 0.9665
emulator_dict['mnu']['default']['N_ur'] = 2.0328
emulator_dict['mnu']['default']['N_ncdm'] = 1
emulator_dict['mnu']['default']['deg_ncdm'] = 1
emulator_dict['mnu']['default']['m_ncdm'] = 0.06
emulator_dict['mnu']['default']['T_cmb'] = 2.7255

emulator_dict['neff']['TT'] = 'TT_neff_v1'
emulator_dict['neff']['TE'] = 'TE_neff_v1'
emulator_dict['neff']['EE'] = 'EE_neff_v1'
emulator_dict['neff']['PP'] = 'PP_neff_v1'
emulator_dict['neff']['PKNL'] = 'PKNL_neff_v1'
emulator_dict['neff']['PKL'] = 'PKL_neff_v1'
emulator_dict['neff']['DER'] = 'DER_neff_v1'
emulator_dict['neff']['DAZ'] = 'DAZ_neff_v1'
emulator_dict['neff']['HZ'] = 'HZ_neff_v1'
emulator_dict['neff']['S8Z'] = 'S8Z_neff_v1'
emulator_dict['neff']['default'] = {}
emulator_dict['neff']['default']['tau_reio'] = 0.054
emulator_dict['neff']['default']['H0'] = 67.66
emulator_dict['neff']['default']['ln10^{10}A_s'] = 3.047
emulator_dict['neff']['default']['omega_b'] = 0.02242
emulator_dict['neff']['default']['omega_cdm'] = 0.11933
emulator_dict['neff']['default']['n_s'] = 0.9665
emulator_dict['neff']['default']['N_ur'] = 2.0328 # this is the default value in class v2 to get Neff = 3.046
emulator_dict['neff']['default']['N_ncdm'] = 1
emulator_dict['neff']['default']['deg_ncdm'] = 1
emulator_dict['neff']['default']['m_ncdm'] = 0.06
emulator_dict['neff']['default']['T_cmb'] = 2.7255

emulator_dict['wcdm']['TT'] = 'TT_w_v1'
emulator_dict['wcdm']['TE'] = 'TE_w_v1'
emulator_dict['wcdm']['EE'] = 'EE_w_v1'
emulator_dict['wcdm']['PP'] = 'PP_w_v1'
emulator_dict['wcdm']['PKNL'] = 'PKNL_w_v1'
emulator_dict['wcdm']['PKL'] = 'PKL_w_v1'
emulator_dict['wcdm']['DER'] = 'DER_w_v1'
emulator_dict['wcdm']['DAZ'] = 'DAZ_w_v1'
emulator_dict['wcdm']['HZ'] = 'HZ_w_v1'
emulator_dict['wcdm']['S8Z'] = 'S8Z_w_v1'
emulator_dict['wcdm']['default'] = {}
emulator_dict['wcdm']['default']['tau_reio'] = 0.054
emulator_dict['wcdm']['default']['H0'] = 67.66
emulator_dict['wcdm']['default']['ln10^{10}A_s'] = 3.047
emulator_dict['wcdm']['default']['omega_b'] = 0.02242
emulator_dict['wcdm']['default']['omega_cdm'] = 0.11933
emulator_dict['wcdm']['default']['n_s'] = 0.9665
emulator_dict['wcdm']['default']['N_ur'] = 2.0328 # this is the default value in class v2 to get Neff = 3.046
emulator_dict['wcdm']['default']['N_ncdm'] = 1
emulator_dict['wcdm']['default']['m_ncdm'] = 0.06
emulator_dict['wcdm']['default']['deg_ncdm'] = 1
emulator_dict['wcdm']['default']['T_cmb'] = 2.7255


emulator_dict['ede']['TT'] = 'TT_v1'
emulator_dict['ede']['TE'] = 'TE_v1'
emulator_dict['ede']['EE'] = 'EE_v1'
emulator_dict['ede']['PP'] = 'PP_v1'
emulator_dict['ede']['PKNL'] = 'PKNL_v1'
emulator_dict['ede']['PKL'] = 'PKL_v1'
emulator_dict['ede']['DER'] = 'DER_v1'
emulator_dict['ede']['DAZ'] = 'DAZ_v1'
emulator_dict['ede']['HZ'] = 'HZ_v1'
emulator_dict['ede']['S8Z'] = 'S8Z_v1'
emulator_dict['ede']['default'] = {}
emulator_dict['ede']['default']['fEDE'] = 0.001
emulator_dict['ede']['default']['tau_reio'] = 0.054
emulator_dict['ede']['default']['H0'] = 67.66
emulator_dict['ede']['default']['ln10^{10}A_s'] = 3.047
emulator_dict['ede']['default']['omega_b'] = 0.02242
emulator_dict['ede']['default']['omega_cdm'] = 0.11933
emulator_dict['ede']['default']['n_s'] = 0.9665
emulator_dict['ede']['default']['log10z_c'] = 3.562 # e.g. from https://github.com/mwt5345/class_ede/blob/master/class/notebooks-ede/2-CMB-Comparison.ipynb
emulator_dict['ede']['default']['thetai_scf'] = 2.83 # e.g. from https://github.com/mwt5345/class_ede/blob/master/class/notebooks-ede/2-CMB-Comparison.ipynb
emulator_dict['ede']['default']['r'] = 0.
emulator_dict['ede']['default']['N_ur'] = 0.00641 # this is the default value in class v2 to get Neff = 3.046
emulator_dict['ede']['default']['N_ncdm'] = 1 ### use equivalence with deg_ncdm = 3
emulator_dict['ede']['default']['deg_ncdm'] = 3 ### use equivalence with deg_ncdm = 3 for faster computation
emulator_dict['ede']['default']['m_ncdm'] = 0.02
emulator_dict['ede']['default']['T_cmb'] = 2.7255

emulator_dict['mnu-3states']['TT'] = 'TT_v1'
emulator_dict['mnu-3states']['TE'] = 'TE_v1'
emulator_dict['mnu-3states']['EE'] = 'EE_v1'
emulator_dict['mnu-3states']['PP'] = 'PP_v1'
emulator_dict['mnu-3states']['PKNL'] = 'PKNL_v1'
emulator_dict['mnu-3states']['PKL'] = 'PKL_v1'
emulator_dict['mnu-3states']['DER'] = 'DER_v1'
emulator_dict['mnu-3states']['DAZ'] = 'DAZ_v1'
emulator_dict['mnu-3states']['HZ'] = 'HZ_v1'
emulator_dict['mnu-3states']['S8Z'] = 'S8Z_v1'
emulator_dict['mnu-3states']['default'] = {}
emulator_dict['mnu-3states']['default']['tau_reio'] = 0.054
emulator_dict['mnu-3states']['default']['H0'] = 67.66
emulator_dict['mnu-3states']['default']['ln10^{10}A_s'] = 3.047
emulator_dict['mnu-3states']['default']['omega_b'] = 0.02242
emulator_dict['mnu-3states']['default']['omega_cdm'] = 0.11933
emulator_dict['mnu-3states']['default']['n_s'] = 0.9665
emulator_dict['mnu-3states']['default']['N_ur'] = 0.00641 # this is the default value in class v2 to get Neff = 3.046
emulator_dict['mnu-3states']['default']['N_ncdm'] = 1 ### use equivalence with deg_ncdm = 3
emulator_dict['mnu-3states']['default']['deg_ncdm'] = 3 ### use equivalence with deg_ncdm = 3 for faster computation
emulator_dict['mnu-3states']['default']['m_ncdm'] = 0.02
emulator_dict['mnu-3states']['default']['T_cmb'] = 2.7255

emulator_dict['ede-v2']['TT'] = 'TT_v2'
emulator_dict['ede-v2']['TE'] = 'TE_v2'
emulator_dict['ede-v2']['EE'] = 'EE_v2'
emulator_dict['ede-v2']['PP'] = 'PP_v2'
emulator_dict['ede-v2']['PKNL'] = 'PKNL_v2'
emulator_dict['ede-v2']['PKL'] = 'PKL_v2'
emulator_dict['ede-v2']['DER'] = 'DER_v2'
emulator_dict['ede-v2']['DAZ'] = 'DAZ_v2'
emulator_dict['ede-v2']['HZ'] = 'HZ_v2'
emulator_dict['ede-v2']['S8Z'] = 'S8Z_v2'

emulator_dict['ede-v2']['default'] = {}
emulator_dict['ede-v2']['default']['fEDE'] = 0.001
emulator_dict['ede-v2']['default']['tau_reio'] = 0.054
emulator_dict['ede-v2']['default']['H0'] = 67.66
emulator_dict['ede-v2']['default']['ln10^{10}A_s'] = 3.047
emulator_dict['ede-v2']['default']['omega_b'] = 0.02242
emulator_dict['ede-v2']['default']['omega_cdm'] = 0.11933
emulator_dict['ede-v2']['default']['n_s'] = 0.9665
emulator_dict['ede-v2']['default']['log10z_c'] = 3.562 # e.g. from https://github.com/mwt5345/class_ede/blob/master/class/notebooks-ede/2-CMB-Comparison.ipynb
emulator_dict['ede-v2']['default']['thetai_scf'] = 2.83 # e.g. from https://github.com/mwt5345/class_ede/blob/master/class/notebooks-ede/2-CMB-Comparison.ipynb
emulator_dict['ede-v2']['default']['r'] = 0.
emulator_dict['ede-v2']['default']['N_ur'] = 0.00441 # this is the default value in class v3 to get Neff = 3.044
emulator_dict['ede-v2']['default']['N_ncdm'] = 1 ### use equivalence with deg_ncdm = 3
emulator_dict['ede-v2']['default']['deg_ncdm'] = 3 ### use equivalence with deg_ncdm = 3 for faster computation
emulator_dict['ede-v2']['default']['m_ncdm'] = 0.02
emulator_dict['ede-v2']['default']['T_cmb'] = 2.7255



def split_emulator_string(input_string):
    match = re.match(r"(.+)-v(\d+)", input_string)
    if match:
        folder = match.group(1)
        version = match.group(2)
        return folder, version
    else:
        folder = input_string
        version = '1'
        return folder, version


