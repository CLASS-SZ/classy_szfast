from .config import path_to_class_sz_data
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from .restore_nn import Restore_NN
from .restore_nn import Restore_PCAplusNN
from .suppress_warnings import suppress_warnings
from .emulators_meta_data import *

from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ
from jax.errors import TracerArrayConversionError


class CosmoPowerJAX_custom(CPJ):
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(verbose=verbose, *args, **kwargs)
        self.ten_to_predictions = True
        if 'ten_to_predictions' in kwargs.keys():
            self.ten_to_predictions = kwargs['ten_to_predictions']
        self._jit_forward = None

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


    def _build_jit_forward(self):
        """Build a JIT-compiled float32 forward pass.

        Casts all weights and normalization stats to float32 JAX arrays and
        captures them in a @jax.jit closure, fusing the entire forward pass
        into a single XLA kernel.  Float32 is appropriate because the NN
        weights are trained in float32; float64 adds overhead with no accuracy
        gain at the ~0.1-1% emulator accuracy level.

        Called lazily on first predict() so that any post-init mutations
        (e.g. ten_to_predictions = False) are captured.
        """
        from jax.nn import sigmoid

        # Cast weights to float32 JAX arrays (captured in the JIT closure)
        weights_f32 = [
            (jnp.array(w, dtype=jnp.float32), jnp.array(b, dtype=jnp.float32))
            for w, b in self.weights
        ]
        hyper_params_f32 = [
            (jnp.array(a, dtype=jnp.float32), jnp.array(b, dtype=jnp.float32))
            for a, b in self.hyper_params
        ]
        param_mean = jnp.array(self.param_train_mean, dtype=jnp.float32)
        param_std = jnp.array(self.param_train_std, dtype=jnp.float32)
        feat_mean = jnp.array(self.feature_train_mean, dtype=jnp.float32)
        feat_std = jnp.array(self.feature_train_std, dtype=jnp.float32)

        n_hidden = len(weights_f32) - 1
        use_full_bias = (self.probe == 'custom_log' or self.probe == 'custom_pca')
        is_log = bool(self.log)
        ten_to_preds = bool(self.ten_to_predictions)

        # PCA probe extras (only needed when is_log is False)
        if not is_log:
            pca_matrix_f32 = jnp.array(self.pca_matrix, dtype=jnp.float32)
            training_std_f32 = jnp.array(self.training_std, dtype=jnp.float32)
            training_mean_f32 = jnp.array(self.training_mean, dtype=jnp.float32)
            is_cmb_pp = (self.probe == 'cmb_pp')
        else:
            pca_matrix_f32 = training_std_f32 = training_mean_f32 = None
            is_cmb_pp = False

        @jax.jit
        def forward(x):
            x = x.astype(jnp.float32)
            # Standardise input
            h = (x - param_mean) / param_std
            # Hidden layers with activation: (beta + sigmoid(alpha*z)*(1-beta)) * z
            for i in range(n_hidden):
                w, b = weights_f32[i]
                alpha, beta = hyper_params_f32[i]
                z = jnp.dot(h, w.T) + b
                h = jnp.multiply(
                    jnp.add(beta, jnp.multiply(
                        sigmoid(jnp.multiply(alpha, z)),
                        jnp.subtract(1.0, beta))),
                    z)
            # Final layer (no activation)
            w, b = weights_f32[-1]
            preds = jnp.dot(h, w.T) + (b if use_full_bias else b[-1])
            # Undo standardisation
            preds = preds * feat_std + feat_mean
            if is_log:
                if ten_to_preds:
                    preds = 10.0 ** preds
            else:
                preds = (preds @ pca_matrix_f32) * training_std_f32 + training_mean_f32
                if is_cmb_pp:
                    preds = 10.0 ** preds
            return preds.squeeze()

        # Warmup: trigger JIT compilation so first real call is fast
        dummy = jnp.zeros((1, self.n_parameters), dtype=jnp.float32)
        forward(dummy).block_until_ready()

        self._jit_forward = forward

    def predict(self, input_vec):
        """Emulate cosmological power spectrum using JIT-compiled forward pass."""
        # Lazy-build JIT forward on first call
        if self._jit_forward is None:
            self._build_jit_forward()

        # Dict-to-array conversion (outside JIT boundary)
        if isinstance(input_vec, dict):
            input_vec = self._dict_to_ordered_arr_np(input_vec)

        if len(input_vec.shape) == 1:
            input_vec = input_vec.reshape(-1, self.n_parameters)

        return self._jit_forward(input_vec)

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


def _load_jax_emulators_for_model(mp):
    """Load all JAX emulator NNs for a single cosmological model."""
    folder, version = split_emulator_string(mp)
    path_to_emulators = path_to_class_sz_data + '/' + folder + '/'

    loaded = {}
    loaded['tt'] = Restore_NN(restore_filename=path_to_emulators + 'TTTEEE/' + emulator_dict[mp]['TT'])
    loaded['te'] = Restore_PCAplusNN(restore_filename=path_to_emulators + 'TTTEEE/' + emulator_dict[mp]['TE'])
    with suppress_warnings():
        loaded['ee'] = Restore_NN(restore_filename=path_to_emulators + 'TTTEEE/' + emulator_dict[mp]['EE'])
    loaded['pp'] = Restore_NN(restore_filename=path_to_emulators + 'PP/' + emulator_dict[mp]['PP'])

    pknl = CosmoPowerJAX_custom(probe='custom_log', filepath=path_to_emulators + 'PK/' + emulator_dict[mp]['PKNL'] + '.npz')
    pknl.ten_to_predictions = False
    loaded['pknl'] = pknl

    pkl = CosmoPowerJAX_custom(probe='custom_log', filepath=path_to_emulators + 'PK/' + emulator_dict[mp]['PKL'] + '.npz')
    pkl.ten_to_predictions = False
    loaded['pkl'] = pkl

    loaded['der'] = CosmoPowerJAX_custom(probe='custom_log', filepath=path_to_emulators + 'derived-parameters/' + emulator_dict[mp]['DER'] + '.npz')

    da = CosmoPowerJAX_custom(probe='custom_log', filepath=path_to_emulators + 'growth-and-distances/' + emulator_dict[mp]['DAZ'] + '.npz')
    if mp != 'ede-v2':
        da.ten_to_predictions = False
    loaded['da'] = da

    loaded['h'] = CosmoPowerJAX_custom(probe='custom_log', filepath=path_to_emulators + 'growth-and-distances/' + emulator_dict[mp]['HZ'] + '.npz')
    loaded['s8'] = Restore_NN(restore_filename=path_to_emulators + 'growth-and-distances/' + emulator_dict[mp]['S8Z'])

    return loaded


class _LazyJaxEmulatorDict(dict):
    """Dict that loads JAX emulators for a cosmological model on first access."""

    def __init__(self, kind):
        super().__init__()
        self._kind = kind

    def __missing__(self, mp):
        if mp not in cosmo_model_list:
            raise KeyError(f"Unknown cosmological model: {mp}")
        if mp not in _loaded_jax_cache:
            _loaded_jax_cache[mp] = _load_jax_emulators_for_model(mp)
        self[mp] = _loaded_jax_cache[mp][self._kind]
        return self[mp]


_loaded_jax_cache = {}

cp_tt_nn_jax = _LazyJaxEmulatorDict('tt')
cp_te_nn_jax = _LazyJaxEmulatorDict('te')
cp_ee_nn_jax = _LazyJaxEmulatorDict('ee')
cp_pp_nn_jax = _LazyJaxEmulatorDict('pp')
cp_pknl_nn_jax = _LazyJaxEmulatorDict('pknl')
cp_pkl_nn_jax = _LazyJaxEmulatorDict('pkl')
cp_der_nn_jax = _LazyJaxEmulatorDict('der')
cp_da_nn_jax = _LazyJaxEmulatorDict('da')
cp_h_nn_jax = _LazyJaxEmulatorDict('h')
cp_s8_nn_jax = _LazyJaxEmulatorDict('s8')
