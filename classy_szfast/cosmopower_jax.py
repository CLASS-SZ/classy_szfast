from .config import path_to_class_sz_data
import numpy as np
import jax.numpy as jnp
from .restore_nn import Restore_NN
from .restore_nn import Restore_PCAplusNN
from .suppress_warnings import suppress_warnings
from .emulators_meta_data import *

from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ
from jax.errors import TracerArrayConversionError


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


class CosmoPowerJAX_custom(CPJ):
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(verbose=verbose, *args, **kwargs)
        self.ten_to_predictions = True
        if 'ten_to_predictions' in kwargs.keys():
            self.ten_to_predictions = kwargs['ten_to_predictions']

    def _dict_to_ordered_arr_np(self,
                               input_dict,
                               ):
        """
        Sort input parameters. Takend verbatim from CP 
        (https://github.com/alessiospuriomancini/cosmopower/blob/main/cosmopower/cosmopower_NN.py#LL291C1-L308C73)

        Parameters:
            input_dict (dict [numpy.ndarray]):
                input dict of (arrays of) parameters to be sorted

        Returns:
            numpy.ndarray:
                parameters sorted according to desired order
        """
        if self.parameters is not None:
            try:
                return np.stack([input_dict[k] for k in self.parameters], axis=1)
            except TracerArrayConversionError:
                converted_dict = {k: jnp.array(v) if isinstance(v, list) else v for k, v in input_dict.items()}
                return jnp.stack([converted_dict[k] for k in self.parameters], axis=1)

        else:
            return np.stack([input_dict[k] for k in input_dict], axis=1)


    def _predict(self, weights, hyper_params, param_train_mean, param_train_std,
                 feature_train_mean, feature_train_std, input_vec):
        """ Forward pass through pre-trained network.
        In its current form, it does not make use of high-level frameworks like
        FLAX et similia; rather, it simply loops over the network layers.
        In future work this can be improved, especially if speed is a problem.
        
        Parameters
        ----------
        weights : array
            The stored weights of the neural network.
        hyper_params : array
            The stored hyperparameters of the activation function for each layer.
        param_train_mean : array
            The stored mean of the training cosmological parameters.
        param_train_std : array
            The stored standard deviation of the training cosmological parameters.
        feature_train_mean : array
            The stored mean of the training features.
        feature_train_std : array
            The stored  standard deviation of the training features.
        input_vec : array of shape (n_samples, n_parameters) or (n_parameters)
            The cosmological parameters given as input to the network.
            
        Returns
        -------
        predictions : array
            The prediction of the trained neural network.
        """        
        act = []
        # Standardise
        layer_out = [(input_vec - param_train_mean)/param_train_std]

        # Loop over layers
        for i in range(len(weights[:-1])):
            w, b = weights[i]
            alpha, beta = hyper_params[i]
            act.append(jnp.dot(layer_out[-1], w.T) + b)
            layer_out.append(self._activation(act[-1], alpha, beta))

        # Final layer prediction (no activations)
        w, b = weights[-1]
        if self.probe == 'custom_log' or self.probe == 'custom_pca':
            # in original CP models, we assumed a full final bias vector...
            preds = jnp.dot(layer_out[-1], w.T) + b
        else:   
            # ... unlike in cpjax, where we used only a single bias vector
            preds = jnp.dot(layer_out[-1], w.T) + b[-1]

        # Undo the standardisation
        preds = preds * feature_train_std + feature_train_mean
        if self.log == True:
            if self.ten_to_predictions:
                preds = 10**preds
        else:
            preds = (preds@self.pca_matrix)*self.training_std + self.training_mean
            if self.probe == 'cmb_pp':
                preds = 10**preds
        predictions = preds.squeeze()
        return predictions

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
    
    cp_pkl_nn_jax[mp] = CosmoPowerJAX_custom(probe='custom_log',filepath=path_to_emulators +'PK/' + emulator_dict[mp]['PKL'] + '.npz')
    cp_pkl_nn_jax[mp].ten_to_predictions = False

    cp_der_nn_jax[mp] = CosmoPowerJAX_custom(probe='custom_log',filepath=path_to_emulators + 'derived-parameters/' + emulator_dict[mp]['DER'] + '.npz')  
    
    cp_da_nn_jax[mp] = CosmoPowerJAX_custom(probe='custom_log',filepath=path_to_emulators + 'growth-and-distances/' + emulator_dict[mp]['DAZ'] + '.npz')
    if mp != 'ede-v2':
        cp_da_nn_jax[mp].ten_to_predictions = False
    # print(cp_da_nn_jax[mp].parameters)
    # print(path_to_emulators + 'growth-and-distances/' + emulator_dict[mp]['HZ'])
    # emulator_custom = CPJ(probe='custom_log',filepath=path_to_emulators + 'growth-and-distances/' + emulator_dict[mp]['HZ'] + '.npz')
    # print(emulator_custom.parameters)
    # exit()

    cp_h_nn_jax[mp] = CosmoPowerJAX_custom(probe='custom_log',filepath=path_to_emulators + 'growth-and-distances/' + emulator_dict[mp]['HZ'] + '.npz')

    cp_s8_nn_jax[mp] = Restore_NN(restore_filename=path_to_emulators + 'growth-and-distances/' + emulator_dict[mp]['S8Z'])
    
