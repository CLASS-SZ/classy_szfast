"""
Pure-numpy ResNet+PCA emulator for the linear matter power spectrum P(k,z).

Architecture: ResNet with 2 residual blocks, 384 hidden dim, 20 PCA components.
Trained on class_sz emulator output (iteration 6 of architecture competition).
Corresponds to cosmo_model='lcdm', tau_reio=0.054, default parameters.

No PyTorch dependency — just numpy and scipy.

Usage:
    from pk_emulator_numpy import PkEmulatorNumpy

    emu = PkEmulatorNumpy('data/iteration6_resnet_pca_numpy.npz')

    # CosmosPower-compatible interface (drop-in for class_sz):
    params_dict = {
        'ln10^{10}A_s': [3.044, 3.044],
        'n_s': [0.965, 0.965],
        'H0': [67.5, 67.5],
        'omega_b': [0.022, 0.022],
        'omega_cdm': [0.12, 0.12],
        'z_pk_save_nonclass': [0.0, 0.5],
    }
    log10pk = emu.predictions_np(params_dict)  # shape (2, 200)
"""

import numpy as np
from scipy.special import erf


def _gelu(x):
    """GELU activation function."""
    return x * 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


class PkEmulatorNumpy:
    """Pure-numpy ResNet+PCA emulator for P(k,z).

    Drop-in replacement for cosmopower Restore_NN for the PKL emulator
    in class_sz (cosmo_model='lcdm').

    Parameters
    ----------
    weights_path : str
        Path to the .npz file with exported weights.
    k_grid : array-like, optional
        Override k-grid. If None, uses the one stored in the weights file.
    """

    def __init__(self, weights_path, k_grid=None):
        w = np.load(weights_path)

        # Preprocessing parameters
        self.X_mean = w['X_mean'].astype(np.float64)
        self.X_std = w['X_std'].astype(np.float64)
        self.y_mean = w['y_mean'].astype(np.float64)
        self.pca_components = w['pca_components'].astype(np.float64)

        # Input projection: Linear(6, 384)
        self.w_input = w['w_input'].astype(np.float64)
        self.b_input = w['b_input'].astype(np.float64)

        # ResBlock 0
        self.w_res0_fc1 = w['w_res0_fc1'].astype(np.float64)
        self.b_res0_fc1 = w['b_res0_fc1'].astype(np.float64)
        self.w_res0_fc2 = w['w_res0_fc2'].astype(np.float64)
        self.b_res0_fc2 = w['b_res0_fc2'].astype(np.float64)

        # ResBlock 1
        self.w_res1_fc1 = w['w_res1_fc1'].astype(np.float64)
        self.b_res1_fc1 = w['b_res1_fc1'].astype(np.float64)
        self.w_res1_fc2 = w['w_res1_fc2'].astype(np.float64)
        self.b_res1_fc2 = w['b_res1_fc2'].astype(np.float64)

        # Output projection: Linear(384, 20)
        self.w_output = w['w_output'].astype(np.float64)
        self.b_output = w['b_output'].astype(np.float64)

        # k-grid
        if k_grid is not None:
            self.k_grid = np.asarray(k_grid, dtype=np.float64)
        elif 'k_grid' in w:
            self.k_grid = w['k_grid'].astype(np.float64)
        else:
            self.k_grid = None

        # CosmosPower-compatible parameter list.
        # Matches the cosmopower PKL emulator for lcdm:
        #   ['ln10^{10}A_s', 'n_s', 'H0', 'omega_b', 'omega_cdm', 'z_pk_save_nonclass']
        # The NN was trained with column order:
        #   [z, omega_b, omega_cdm, H0, ln10^{10}A_s, n_s]
        # So we reorder in predictions_np.
        self.parameters = ['ln10^{10}A_s', 'n_s', 'H0', 'omega_b', 'omega_cdm', 'z_pk_save_nonclass']

        # Internal column order for the NN
        self._nn_columns = ['z_pk_save_nonclass', 'omega_b', 'omega_cdm', 'H0', 'ln10^{10}A_s', 'n_s']

    def predict(self, X):
        """Forward pass: predict log10(P(k)) spectra.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, 6)
            Columns in NN order: [z, omega_b, omega_cdm, H0, ln10^{10}A_s, n_s]

        Returns
        -------
        np.ndarray, shape (n_samples, 200)
            Predicted log10(P(k)) at the k-grid points.
        """
        X = np.atleast_2d(X).astype(np.float64)

        # Input standardization
        x = (X - self.X_mean) / self.X_std

        # Input projection + GELU
        x = x @ self.w_input.T + self.b_input
        x = _gelu(x)

        # ResBlock 0
        residual = x
        x = x @ self.w_res0_fc1.T + self.b_res0_fc1
        x = _gelu(x)
        x = x @ self.w_res0_fc2.T + self.b_res0_fc2
        x = _gelu(x)
        x = residual + x

        # ResBlock 1
        residual = x
        x = x @ self.w_res1_fc1.T + self.b_res1_fc1
        x = _gelu(x)
        x = x @ self.w_res1_fc2.T + self.b_res1_fc2
        x = _gelu(x)
        x = residual + x

        # Output projection → PCA coefficients → reconstruction
        coeffs = x @ self.w_output.T + self.b_output
        return self.y_mean + coeffs @ self.pca_components

    def predictions_np(self, params_dict):
        """CosmosPower-compatible interface (drop-in for Restore_NN).

        Parameters
        ----------
        params_dict : dict
            Keys matching cosmopower convention. Required keys:
            'ln10^{10}A_s', 'n_s', 'H0', 'omega_b', 'omega_cdm', 'z_pk_save_nonclass'
            Values are lists/arrays of length n_samples.
            Extra keys (tau_reio, m_ncdm, N_ur, etc.) are ignored.

        Returns
        -------
        np.ndarray, shape (n_samples, 200)
            Predicted log10(P(k)).
        """
        # Reorder from cosmopower dict keys to NN column order
        X = np.stack([np.asarray(params_dict[k], dtype=np.float64)
                       for k in self._nn_columns], axis=1)
        return self.predict(X)
