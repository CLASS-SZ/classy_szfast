from .config import path_to_class_sz_data
import numpy as np
from .restore_nn import Restore_NN
from .restore_nn import Restore_PCAplusNN
from .suppress_warnings import suppress_warnings
from .emulators_meta_data import *


def _load_emulators_for_model(mp):
    """Load all emulator NNs for a single cosmological model."""
    folder, version = split_emulator_string(mp)
    path_to_emulators = path_to_class_sz_data + '/' + folder + '/'

    loaded = {}
    loaded['tt'] = Restore_NN(restore_filename=path_to_emulators + 'TTTEEE/' + emulator_dict[mp]['TT'])
    loaded['te'] = Restore_PCAplusNN(restore_filename=path_to_emulators + 'TTTEEE/' + emulator_dict[mp]['TE'])
    with suppress_warnings():
        loaded['ee'] = Restore_NN(restore_filename=path_to_emulators + 'TTTEEE/' + emulator_dict[mp]['EE'])
    loaded['pp'] = Restore_NN(restore_filename=path_to_emulators + 'PP/' + emulator_dict[mp]['PP'])
    loaded['pknl'] = Restore_NN(restore_filename=path_to_emulators + 'PK/' + emulator_dict[mp]['PKNL'])
    loaded['pkl'] = Restore_NN(restore_filename=path_to_emulators + 'PK/' + emulator_dict[mp]['PKL'])

    if (mp == 'lcdm') and (dofftlog_alphas == True):
        loaded['pkl_fftlog_real'] = Restore_PCAplusNN(
            restore_filename=path_to_emulators + 'PK/' + emulator_dict[mp]['PKLFFTLOG_ALPHAS_REAL'])
        loaded['pkl_fftlog_imag'] = Restore_PCAplusNN(
            restore_filename=path_to_emulators + 'PK/' + emulator_dict[mp]['PKLFFTLOG_ALPHAS_IMAG'])
        loaded['pkl_fftlog_nus'] = np.load(path_to_emulators + 'PK/PKL_FFTLog_alphas_nu_v1.npz')

    loaded['der'] = Restore_NN(restore_filename=path_to_emulators + 'derived-parameters/' + emulator_dict[mp]['DER'])
    loaded['da'] = Restore_NN(restore_filename=path_to_emulators + 'growth-and-distances/' + emulator_dict[mp]['DAZ'])
    loaded['h'] = Restore_NN(restore_filename=path_to_emulators + 'growth-and-distances/' + emulator_dict[mp]['HZ'])
    loaded['s8'] = Restore_NN(restore_filename=path_to_emulators + 'growth-and-distances/' + emulator_dict[mp]['S8Z'])

    return loaded


class _LazyEmulatorDict(dict):
    """Dict that loads emulators for a cosmological model on first access."""

    def __init__(self, kind):
        super().__init__()
        self._kind = kind          # e.g. 'pkl', 'tt', 'h', ...
        self._loaded_models = {}   # shared cache across all LazyEmulatorDicts

    def __missing__(self, mp):
        if mp not in cosmo_model_list:
            raise KeyError(f"Unknown cosmological model: {mp}")
        # Load ALL emulators for this model (once per model, shared across dicts)
        if mp not in _loaded_cache:
            _loaded_cache[mp] = _load_emulators_for_model(mp)
        self[mp] = _loaded_cache[mp][self._kind]
        return self[mp]


# Shared cache: loaded once per model, reused by all lazy dicts
_loaded_cache = {}

cp_tt_nn = _LazyEmulatorDict('tt')
cp_te_nn = _LazyEmulatorDict('te')
cp_ee_nn = _LazyEmulatorDict('ee')
cp_pp_nn = _LazyEmulatorDict('pp')
cp_pknl_nn = _LazyEmulatorDict('pknl')
cp_pkl_nn = _LazyEmulatorDict('pkl')
cp_pkl_fftlog_alphas_real_nn = _LazyEmulatorDict('pkl_fftlog_real')
cp_pkl_fftlog_alphas_imag_nn = _LazyEmulatorDict('pkl_fftlog_imag')
cp_pkl_fftlog_alphas_nus = _LazyEmulatorDict('pkl_fftlog_nus')
cp_der_nn = _LazyEmulatorDict('der')
cp_da_nn = _LazyEmulatorDict('da')
cp_h_nn = _LazyEmulatorDict('h')
cp_s8_nn = _LazyEmulatorDict('s8')
