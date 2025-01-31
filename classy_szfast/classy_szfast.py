from .utils import *
from .utils import Const, jax_gradient
from .config import *
import numpy as np
from .emulators_meta_data import emulator_dict, dofftlog_alphas, cp_l_max_scalars, cosmo_model_list
from .cosmopower import cp_tt_nn, cp_te_nn, cp_ee_nn, cp_ee_nn, cp_pp_nn, cp_pknl_nn, cp_pkl_nn, cp_der_nn, cp_da_nn, cp_h_nn, cp_s8_nn, cp_pkl_fftlog_alphas_real_nn, cp_pkl_fftlog_alphas_imag_nn, cp_pkl_fftlog_alphas_nus
from .cosmopower_jax import cp_tt_nn_jax, cp_te_nn_jax, cp_ee_nn_jax, cp_ee_nn_jax, cp_pp_nn_jax, cp_pknl_nn_jax, cp_pkl_nn_jax, cp_der_nn_jax, cp_da_nn_jax, cp_h_nn_jax, cp_s8_nn_jax
from .pks_and_sigmas import *
import scipy
import time
from multiprocessing import Process
from mcfit import TophatVar
from scipy.interpolate import CubicSpline
import pickle
import jax.numpy as jnp
import jax.scipy as jscipy

H_units_conv_factor = {"1/Mpc": 1, "km/s/Mpc": Const.c_km_s}


import logging

def configure_logging(level=logging.INFO):
    logging.basicConfig(
        format='%(levelname)s - %(message)s',
        level=level
    )

def set_verbosity(verbosity):
    levels = {
        'none': logging.CRITICAL,
        'minimal': logging.INFO,
        'extensive': logging.DEBUG
    }
    # print(f'Setting verbosity to {verbosity}')
    level = levels.get(verbosity, logging.INFO)
    configure_logging(level)



def update_params_with_defaults(params_values, default_values):
    """
    Update params_values with default values if they don't already exist.

    Args:
    params_values (dict): Dictionary containing parameter values.
    default_values (dict): Dictionary containing default parameter values.
    """

    
    # Update params_values with default values if key does not exist
    for key, value in default_values.items():
        if key not in params_values:
            params_values[key] = value


class Class_szfast(object):
    def __init__(self,
                 params_settings = {},
                #lowring=False,  some options if needed
                 **kwargs):
        # some parameters
        # self.xy = xy
        # self.lowring = lowring


        self.A_s_fast = 0 
        self.logA_fast = 0


        try:
            set_verbosity(params_settings["classy_sz_verbose"])
        except:
            pass
        self.logger = logging.getLogger(__name__)


        self.jax_mode = params_settings["jax"]

 
        # print(f"JAX mode: {self.jax_mode}")
        

        # cosmopower emulators
        # self.cp_path_to_cosmopower_organization = path_to_cosmopower_organization + '/'
        self.cp_tt_nn = cp_tt_nn
        self.cp_te_nn = cp_te_nn
        self.cp_ee_nn = cp_ee_nn
        self.cp_pp_nn = cp_pp_nn
        self.cp_pknl_nn  = cp_pknl_nn

       
        self.cp_lmax = cp_l_max_scalars

        cosmo_model_dict = {i: model for i, model in enumerate(cosmo_model_list)}

        if (cosmo_model_dict[params_settings['cosmo_model']] == 'ede-v2'):
        
            self.cszfast_pk_grid_zmax = 20.
            self.cszfast_pk_grid_kmin = 5e-4
            self.cszfast_pk_grid_kmax = 10. 
            self.cp_kmax = self.cszfast_pk_grid_kmax
            self.cp_kmin = self.cszfast_pk_grid_kmin
            # self.logger.info(f">>> using kmin = {self.cp_kmin}")
            # self.logger.info(f">>> using kmax = {self.cp_kmax}")
            # self.logger.info(f">>> using zmax = {self.cszfast_pk_grid_zmax}")
        
        else:
        
            self.cszfast_pk_grid_zmax = 5. #  max z of our pk emulators (sept 23)
            self.cszfast_pk_grid_kmin = 1e-4
            self.cszfast_pk_grid_kmax = 50.
            self.cp_kmax = self.cszfast_pk_grid_kmax
            self.cp_kmin = self.cszfast_pk_grid_kmin
            # self.logger.info(f">>> using kmin = {self.cp_kmin}")
            # self.logger.info(f">>> using kmax = {self.cp_kmax}")
            # self.logger.info(f">>> using zmax = {self.cszfast_pk_grid_zmax}")
        
        if self.jax_mode:
            self.cp_h_nn = cp_h_nn_jax
            self.cp_da_nn = cp_da_nn_jax
            self.cp_pkl_nn = cp_pkl_nn_jax
            self.cp_der_nn = cp_der_nn_jax

            self.pi = jnp.pi
            self.transpose = jnp.transpose
            self.asarray = jnp.asarray
            self.log = jnp.log
            self.pow = jnp.power

            self.sigma_B = 2. * self.pow(self.pi,5) * self.pow(Const._k_B_,4) / 15. / self.pow(Const._h_P_,3) / self.pow(Const._c_,2)

            self.linspace = jnp.linspace
            self.geomspace = jnp.geomspace
            self.arange = jnp.arange
            self.zeros = jnp.zeros
            self.gradient = jax_gradient


        else:
            self.cp_h_nn = cp_h_nn
            self.cp_da_nn = cp_da_nn
            self.cp_pkl_nn = cp_pkl_nn
            self.cp_der_nn = cp_der_nn
            self.pi = np.pi
            self.transpose = np.transpose
            self.pow = np.power

            self.sigma_B = 2. * self.pow(self.pi,5) * self.pow(Const._k_B_,4) / 15. / self.pow(Const._h_P_,3) / self.pow(Const._c_,2)


            self.linspace = np.linspace
            self.geomspace = np.geomspace
            self.arange = np.arange
            self.zeros = np.zeros
            self.asarray = np.asarray
            self.log = np.log
            self.gradient = np.gradient


        self.cp_ls = self.arange(2,self.cp_lmax+1)
        self.cp_predicted_tt_spectrum =self.zeros(self.cp_lmax)
        self.cp_predicted_te_spectrum =self.zeros(self.cp_lmax)
        self.cp_predicted_ee_spectrum =self.zeros(self.cp_lmax)
        self.cp_predicted_pp_spectrum =self.zeros(self.cp_lmax)

        self.cp_s8_nn = cp_s8_nn

        self.emulator_dict = emulator_dict

        if dofftlog_alphas == True:
            self.cp_pkl_fftlog_alphas_nus = cp_pkl_fftlog_alphas_nus
            self.cp_pkl_fftlog_alphas_real_nn  = cp_pkl_fftlog_alphas_real_nn
            self.cp_pkl_fftlog_alphas_imag_nn = cp_pkl_fftlog_alphas_imag_nn

        self.cosmo_model = 'ede-v2'

        self.use_Amod = 0
        self.Amod = 0 
        
        self.use_pk_z_bins = 0
        self.pk_z_bins_z1 = 0
        self.pk_z_bins_z2 = 0
        self.pk_z_bins_A0 = 0
        self.pk_z_bins_A1 = 0
        self.pk_z_bins_A2 = 0
        

        if cosmo_model_dict[params_settings['cosmo_model']] == 'ede-v2':

            self.cp_ndspl_k = 1
            self.cp_nk = 1000
        
        else:
        
            self.cp_ndspl_k = 10
            self.cp_nk = 5000


        self.cszfast_ldim = 20000 # used for the cls arrays
        self.cszfast_pk_grid_nz = 100 # has to be same as narraySZ, i.e., ndim_redshifts; it is setup hereafter if ndim_redshifts is passed
        

        self.cszfast_pk_grid_z = self.linspace(0.,self.cszfast_pk_grid_zmax,self.cszfast_pk_grid_nz)
        self.cszfast_pk_grid_ln1pz = self.log(1.+self.cszfast_pk_grid_z)


        self.cszfast_pk_grid_k = self.geomspace(self.cp_kmin,self.cp_kmax,self.cp_nk)[::self.cp_ndspl_k]
        
        self.cszfast_pk_grid_lnk = self.log(self.cszfast_pk_grid_k)
        
        self.cszfast_pk_grid_nk = len(self.geomspace(self.cp_kmin,self.cp_kmax,self.cp_nk)[::self.cp_ndspl_k]) # has to be same as ndimSZ, and the same as dimension of cosmopower pk emulators
        
        for k,v in params_settings.items():

            if k == 'ndim_redshifts':
                
                self.cszfast_pk_grid_nz = v
                self.cszfast_pk_grid_z = self.linspace(0.,self.cszfast_pk_grid_zmax,self.cszfast_pk_grid_nz)
                self.cszfast_pk_grid_ln1pz = self.log(1.+self.cszfast_pk_grid_z)

                self.cszfast_pk_grid_pknl_flat = self.zeros(self.cszfast_pk_grid_nz*self.cszfast_pk_grid_nk)
                self.cszfast_pk_grid_pkl_flat = self.zeros(self.cszfast_pk_grid_nz*self.cszfast_pk_grid_nk)

            if k == 'cosmo_model':

                self.cosmo_model = cosmo_model_dict[v]

            if k == 'use_Amod':

                self.use_Amod = v
                self.Amod  = params_settings['Amod']

            if k == 'use_pk_z_bins':
                self.use_pk_z_bins = v
                self.pk_z_bins_z1 = params_settings['pk_z_bins_z1']
                self.pk_z_bins_z2 = params_settings['pk_z_bins_z2']
                self.pk_z_bins_A0 = params_settings['pk_z_bins_A0']
                self.pk_z_bins_A1 = params_settings['pk_z_bins_A1']
                self.pk_z_bins_A2 = params_settings['pk_z_bins_A2']




        if cosmo_model_dict[params_settings['cosmo_model']] == 'ede-v2':
        
            self.pk_power_fac = self.cszfast_pk_grid_k**-3
        
        else:

            ls = self.arange(2,self.cp_nk+2)[::self.cp_ndspl_k] # jan 10 ndspl
            dls = ls*(ls+1.)/2./self.pi
            self.pk_power_fac= (dls)**-1


        self.cp_z_interp = self.linspace(0.,20.,5000)

        self.csz_base = None


        self.cszfast_zgrid_zmin = 0.
        self.cszfast_zgrid_zmax = 4.
        self.cszfast_zgrid_nz = 250
        self.cszfast_zgrid = self.linspace(self.cszfast_zgrid_zmin,
                                         self.cszfast_zgrid_zmax,
                                         self.cszfast_zgrid_nz)


        self.cszfast_mgrid_mmin = 1e10
        self.cszfast_mgrid_mmax = 1e15
        self.cszfast_mgrid_nm = 50
        self.cszfast_mgrid = self.geomspace(self.cszfast_mgrid_mmin,
                                          self.cszfast_mgrid_mmax,
                                          self.cszfast_mgrid_nm)

        self.cszfast_gas_pressure_xgrid_xmin = 1e-2
        self.cszfast_gas_pressure_xgrid_xmax = 1e2
        self.cszfast_gas_pressure_xgrid_nx = 100
        self.cszfast_gas_pressure_xgrid = self.geomspace(self.cszfast_gas_pressure_xgrid_xmin,
                                                       self.cszfast_gas_pressure_xgrid_xmax,
                                                       self.cszfast_gas_pressure_xgrid_nx)
        
        self.params_for_emulators = {}


    def get_all_relevant_params(self,params_values_dict=None):
        if params_values_dict:
            params_values = params_values_dict.copy()
        else:
            params_values = self.params_for_emulators
        update_params_with_defaults(params_values, self.emulator_dict[self.cosmo_model]['default'])
        params_values['h'] = params_values['H0']/100.
        params_values['Omega_b'] = params_values['omega_b']/params_values['h']**2.
        params_values['Omega_cdm'] = params_values['omega_cdm']/params_values['h']**2.
        params_values['Omega0_g'] = (4.*self.sigma_B/Const._c_*pow(params_values['T_cmb'],4.)) / (3.*Const._c_*Const._c_*1.e10*params_values['h']*params_values['h']/Const._Mpc_over_m_/Const._Mpc_over_m_/8./self.pi/Const._G_)
        params_values['Omega0_ur'] = params_values['N_ur']*7./8.*self.pow(4./11.,4./3.)*params_values['Omega0_g']
        params_values['Omega0_ncdm'] = params_values['deg_ncdm']*params_values['m_ncdm']/(93.14*params_values['h']*params_values['h']) ## valid only in standard cases, default T_ncdm etc
        params_values['Omega_Lambda'] = 1. - params_values['Omega0_g'] - params_values['Omega_b'] - params_values['Omega_cdm'] - params_values['Omega0_ncdm'] - params_values['Omega0_ur']
        params_values['Omega0_m'] = params_values['Omega_cdm'] + params_values['Omega_b'] + params_values['Omega0_ncdm']
        params_values['Omega0_r'] = params_values['Omega0_ur']+params_values['Omega0_g']
        params_values['Omega0_m_nonu'] = params_values['Omega0_m'] - params_values['Omega0_ncdm']
        params_values['Omega0_cb'] = params_values['Omega0_m_nonu'] 
        H0 = params_values['H0']*1.e3/Const._c_
        params_values['Rho_crit_0'] = (3./(8.*self.pi*Const._G_*Const._M_sun_))*pow(Const._Mpc_over_m_,1)*self.pow(Const._c_,2)*self.pow(H0,2)/self.pow(params_values['h'],2)
        return params_values


    def get_rho_crit_at_z(self,z,params_values_dict=None):
        params_values = self.get_all_relevant_params(params_values_dict)
        self.calculate_hubble(**params_values_dict)
        H = self.hz_interp(z)
        rho_crit = (3./(8.*self.pi*Const._G_*Const._M_sun_))*pow(Const._Mpc_over_m_,1)*self.pow(Const._c_,2)*self.pow(H,2)/self.pow(params_values['h'],2)
        return rho_crit
    
    def get_volume_dVdzdOmega_at_z(self,z,params_values_dict=None):
        params_values = self.get_all_relevant_params(params_values_dict)
        self.calculate_hubble(**params_values_dict)
        Ez = self.hz_interp(z)/self.hz_interp(0.)
        self.calculate_chi(**params_values_dict)
        dA = self.chi_interp(z)/(1.+z)
        rz = dA*(1.+z)*params_values['h']
        dVdzdOmega = 2.99792458e8/1.0e5*rz*rz/Ez
        return dVdzdOmega
    
    def get_sigma8_and_der(self,params_values_dict=None):
        params_values = self.get_all_relevant_params(params_values_dict)
        self.calculate_sigma8_and_der(**params_values)
        return self.cp_predicted_der

    def get_r_delta_of_m_delta_at_z(self,delta,m_delta,z,params_values_dict=None):
        if params_values_dict:
            params_values = params_values_dict.copy()
        else:
            params_values = self.params_for_emulators
        rho_crit = self.get_rho_crit_at_z(z,params_values_dict=params_values_dict)
        return (m_delta*3./4./self.pi/delta/rho_crit)**(1./3.)

    def get_nu_at_z_and_m(self,z,m,params_values_dict=None):
        if params_values_dict:
            params_values = params_values_dict.copy()
        else:
            params_values = self.params_for_emulators

        delta_c =  (3./20.)*self.pow(12.*self.pi,2./3.) # this is = 1.686470199841145
        # note here we dont use matter dependent delta_c
        # which would be multiplied by (1.+0.012299*log10(pvecback[pba->index_bg_Omega_m]));
        sigma = 1. 
        return (delta_c/sigma)**2 


    def find_As(self,params_cp):

        sigma_8_asked = params_cp["sigma8"]

        update_params_with_defaults(params_cp, self.emulator_dict[self.cosmo_model]['default'])

        def to_root(ln10_10_As_goal):
            params_cp["ln10^{10}A_s"] = ln10_10_As_goal[0]
            params_dict = {}
            for k,v in params_cp.items():
                params_dict[k]=[v]
            
            return self.cp_der_nn[self.cosmo_model].ten_to_predictions_np(params_dict)[0][1]-sigma_8_asked

        lnA_s = optimize.root(to_root,
                              x0=3.046,
                              #tol = 1e-10,
                              method="hybr")
        
        params_cp['ln10^{10}A_s'] = lnA_s.x[0]

        params_cp.pop('sigma8')

        return 1


    def get_H0_from_thetas(self,params_values):

        update_params_with_defaults(params_values, self.emulator_dict[self.cosmo_model]['default'])

        # print(params_values)
        theta_s_asked = params_values['100*theta_s']
        def fzero(H0_goal):
          params_values['H0'] = H0_goal[0]
          params_dict = {}
          for k,v in params_values.items():
              params_dict[k]=[v]

          predicted_der_params = self.cp_der_nn[self.cosmo_model].ten_to_predictions_np(params_dict)
          return predicted_der_params[0][0]-theta_s_asked
        sol = optimize.root(fzero,
                          #[40., 99.],
                          x0 = 100.*(3.54*theta_s_asked**2-5.455*theta_s_asked+2.548),
                          #jac=jac,
                          tol = 1e-10,
                          method='hybr')

        params_values.pop('100*theta_s')
        params_values['H0'] = sol.x[0]
        return 1


    def calculate_pkl_fftlog_alphas(self,zpk = 0.,**params_values_dict):
        params_values = params_values_dict.copy()
        params_dict = {}
        for k,v in params_values.items():
            params_dict[k]=[v]
        params_dict['z_pk_save_nonclass'] = [zpk]

        predicted_testing_alphas_creal = self.cp_pkl_fftlog_alphas_real_nn[self.cosmo_model].predictions_np(params_dict)[0]
        predicted_testing_alphas_cimag = self.cp_pkl_fftlog_alphas_imag_nn[self.cosmo_model].predictions_np(params_dict)[0]
        predicted_testing_alphas_cimag = np.append(predicted_testing_alphas_cimag,0.)
        creal = predicted_testing_alphas_creal
        cimag = predicted_testing_alphas_cimag
        Nmax = len(self.cszfast_pk_grid_k)
        cnew = self.zeros(Nmax+1,dtype=complex)
        for i in range(Nmax+1):
            if i<int(Nmax/2):
                cnew[i] = complex(creal[i],cimag[i])
            elif i==int(Nmax/2):
                cnew[i] = complex(creal[i],cimag[i])
            else:
                j = i-int(Nmax/2)
                cnew[i] = complex(creal[::-1][j],-cimag[::-1][j])
        # self.predicted_fftlog_pkl_alphas = cnew
        return cnew

    def get_pkl_reconstructed_from_fftlog(self,zpk = 0.,**params_values_dict):
        #c_n_math = self.predicted_fftlog_pkl_alphas
        c_n_math = self.calculate_pkl_fftlog_alphas(zpk = zpk,**params_values_dict)
        nu_n_math = self.cp_pkl_fftlog_alphas_nus[self.cosmo_model]['arr_0']
        Nmax = int(len(c_n_math)-1)
        term1 = c_n_math[int(Nmax/2)]*self.cszfast_pk_grid_k**(nu_n_math[int(Nmax/2)])
        term2_array = [c_n_math[int(Nmax/2)+i]*self.cszfast_pk_grid_k**(nu_n_math[int(Nmax/2)+i]) for i in range(1, int(Nmax/2)+1)]
        pk_reconstructed = (term1 + 2*np.sum(term2_array,axis=0)).real
        return self.cszfast_pk_grid_k,pk_reconstructed

    def calculate_cmb(self,
                      want_tt=True,
                      want_te=True,
                      want_ee=True,
                      want_pp=1,
                      **params_values_dict):
        

        params_values = params_values_dict.copy()

        update_params_with_defaults(params_values, self.emulator_dict[self.cosmo_model]['default'])

        params_dict = {}

        for k,v in params_values.items():
            params_dict[k]=[v]

        if 'm_ncdm' in params_dict.keys():
            if isinstance(params_dict['m_ncdm'][0],str): 
                params_dict['m_ncdm'] =  [float(params_dict['m_ncdm'][0].split(',')[0])]



        # if want_tt: for now always want_tt = True
        self.cp_predicted_tt_spectrum = self.cp_tt_nn[self.cosmo_model].ten_to_predictions_np(params_dict)[0]

        nl = len(self.cp_predicted_tt_spectrum)
        cls = {}
        cls['ell'] = self.arange(20000)
        cls['tt'] = self.zeros(20000)
        cls['te'] = self.zeros(20000)
        cls['ee'] = self.zeros(20000)
        cls['pp'] = self.zeros(20000)
        cls['bb'] = self.zeros(20000)
        lcp = self.asarray(cls['ell'][2:nl+2])

        # print('cosmo_model:',self.cosmo_model,nl)
        if self.cosmo_model == 'ede-v2':
            factor_ttteee = 1./lcp**2 
            factor_pp = 1./lcp**3
        else:
            factor_ttteee = 1./(lcp*(lcp+1.)/2./self.pi)
            factor_pp = 1./(lcp*(lcp+1.))**2.        

        self.cp_predicted_tt_spectrum *= factor_ttteee
        

        if want_te:
            self.cp_predicted_te_spectrum = self.cp_te_nn[self.cosmo_model].predictions_np(params_dict)[0]
            self.cp_predicted_te_spectrum *= factor_ttteee
        if want_ee:
            self.cp_predicted_ee_spectrum = self.cp_ee_nn[self.cosmo_model].ten_to_predictions_np(params_dict)[0]
            self.cp_predicted_ee_spectrum *= factor_ttteee
        if want_pp:
            self.cp_predicted_pp_spectrum = self.cp_pp_nn[self.cosmo_model].ten_to_predictions_np(params_dict)[0]
            self.cp_predicted_pp_spectrum *= factor_pp

        

        
        # print('>>> clssy_szfast.py cmb computed')

    def load_cmb_cls_from_file(self,**params_values_dict):
        cls_filename = params_values_dict['cmb_cls_filename']
        with open(cls_filename, 'rb') as handle:
            cmb_cls_loaded = pickle.load(handle)
        nl_cls_file = len(cmb_cls_loaded['ell'])
        cls_ls = cmb_cls_loaded['ell']
        dlfac = cls_ls*(cls_ls+1.)/2./self.pi
        cls_tt = cmb_cls_loaded['tt']*dlfac
        cls_te = cmb_cls_loaded['te']*dlfac
        cls_ee = cmb_cls_loaded['ee']*dlfac
        # cls_pp = cmb_cls_loaded['pp']*dlfac # check the normalization.
        nl_cp_cmb = len(self.cp_predicted_tt_spectrum)
        nl_req = min(nl_cls_file,nl_cp_cmb)

        self.cp_predicted_tt_spectrum[:nl_req-2] = cls_tt[2:nl_req]
        self.cp_predicted_te_spectrum[:nl_req-2] = cls_te[2:nl_req]
        self.cp_predicted_ee_spectrum[:nl_req-2] = cls_ee[2:nl_req]
        # self.cp_predicted_pp_spectrum[:nl_req-2] = cls_pp[2:nl_req]


    def calculate_pkl(self,
                      # cosmo_model = self.cosmo_model,
                      **params_values_dict):
 
        z_arr = self.cszfast_pk_grid_z
        k_arr = self.cszfast_pk_grid_k 

        # print(">>> z_arr:",z_arr)
        # print(">>> k_arr:",k_arr)
        # import sys
        


        params_values = params_values_dict.copy()
        update_params_with_defaults(params_values, self.emulator_dict[self.cosmo_model]['default'])


        params_dict = {}
        for k,v in zip(params_values.keys(),params_values.values()):
            params_dict[k]=[v]

        if 'm_ncdm' in params_dict.keys():
            if isinstance(params_dict['m_ncdm'][0],str): 
                params_dict['m_ncdm'] =  [float(params_dict['m_ncdm'][0].split(',')[0])]



        predicted_pk_spectrum_z = []

        if self.use_Amod:

            for zp in z_arr:

                params_dict_pp = params_dict.copy()
                params_dict_pp['z_pk_save_nonclass'] = [zp]
                pkl_p = self.cp_pkl_nn[self.cosmo_model].predictions_np(params_dict_pp)[0]
                pknl_p = self.cp_pknl_nn[self.cosmo_model].predictions_np(params_dict_pp)[0]
                pk_ae  = pkl_p + self.Amod*(pknl_p-pkl_p)
                predicted_pk_spectrum_z.append(pk_ae)

        elif self.use_pk_z_bins:
            # print('>>> using pk_z_bins')
            for zp in z_arr:
                params_dict_pp = params_dict.copy()
                params_dict_pp['z_pk_save_nonclass'] = [zp]
                pkl_p = self.cp_pkl_nn[self.cosmo_model].predictions_np(params_dict_pp)[0]
                if zp < self.pk_z_bins_z1:
                    pklp = self.pk_z_bins_A0*pkl_p
                elif zp < self.pk_z_bins_z2:
                    pklp = self.pk_z_bins_A1*pkl_p
                else:
                    pklp = self.pk_z_bins_A2*pkl_p
                predicted_pk_spectrum_z.append(pklp)
        else:

            for zp in z_arr:
            
                params_dict_pp = params_dict.copy()
                params_dict_pp['z_pk_save_nonclass'] = [zp]
                if self.jax_mode:
                    predicted_pk_spectrum_z.append(self.cp_pkl_nn[self.cosmo_model].predict(params_dict_pp))
                else:
                    predicted_pk_spectrum_z.append(self.cp_pkl_nn[self.cosmo_model].predictions_np(params_dict_pp)[0])

                # if abs(zp-0.5) < 0.01:
                #   print(">>> predicted_pk_spectrum_z:",predicted_pk_spectrum_z[-1])
                #   import pprint
                #   pprint.pprint(params_dict_pp)
       
        predicted_pk_spectrum = self.asarray(predicted_pk_spectrum_z)


        pk = 10.**predicted_pk_spectrum

        pk_re = pk*self.pk_power_fac
        pk_re = self.transpose(pk_re)

        # print(">>> pk_re:",pk_re)
        # import sys
        # sys.exit(0)

        if self.jax_mode:
            self.pkl_interp = None
        else:
            self.pkl_interp = PowerSpectrumInterpolator(z_arr,k_arr,self.log(pk_re).T,logP=True)

        self.cszfast_pk_grid_pk = pk_re
        self.cszfast_pk_grid_pkl_flat = pk_re.flatten()

        return pk_re, k_arr, z_arr


    def calculate_sigma(self,**params_values_dict):

        params_values = params_values_dict.copy()

        k = self.cszfast_pk_grid_k

        P = self.cszfast_pk_grid_pk

        var = P.copy()

        dvar = P.copy()
        # dvar_grads = P.copy()

        for iz,zp in enumerate(self.cszfast_pk_grid_z):

            R, var[:,iz] = TophatVar(k, lowring=True)(P[:,iz], extrap=True)

            dvar[:,iz] = self.gradient(var[:,iz], R) ## old form with jax gradient from inigo zubeldia if needed. 
            # _, dvar[:,iz] = TophatVar(k, lowring=True, deriv=1)(P[:,iz]*k, extrap=True) # new form doesnt seem to work accurately

        # dvar = dvar/(2.*np.sqrt(var))
        # print(dvar_grads/dvar)
        # print(k)
        # print(R)
        # print(k*R)
        # exit(0)
 

        self.cszfast_pk_grid_lnr = self.log(R)
        self.cszfast_pk_grid_sigma2 = var
 
        self.cszfast_pk_grid_sigma2_flat = var.flatten()
        self.cszfast_pk_grid_lnsigma2_flat = 0.5*self.log(var.flatten())
 
        self.cszfast_pk_grid_dsigma2 = dvar
        self.cszfast_pk_grid_dsigma2_flat = dvar.flatten()
 
        return 0
    
    def calculate_vrms2(self,**params_values_dict):
        # k = self.cszfast_pk_grid_k
        k = np.geomspace(1e-5,1e1,1000)
        
        
        # z = self.cszfast_pk_grid_z
        z = np.linspace(0,3,800)

        # P = self.cszfast_pk_grid_pk
        P = np.asarray([self.get_pkl_at_k_and_z(k,zpx) for zpx in z])

        a = 1/(1+z)
        H = self.get_hubble(z)
        z0 = 0.

        k0 = 1e-2
        pks = self.get_pkl_at_k_and_z(k0,z)
        pks0 = self.get_pkl_at_k_and_z(k0,z0)
        D = np.sqrt(pks/pks0)

        f = -np.gradient(np.log(D), np.log(1 + z))

        W = f*a*H
        x = np.log(k)
        I = 1/3*W[:,None]**2*P*k/2/np.pi**2


        vrms2 = scipy.integrate.simpson(I,x=x,axis=1)


        self.cszfast_3c2vrms2 = np.interp(self.cszfast_pk_grid_z,z,np.log(vrms2*3.*Const._c_**2*1e-6))
        return 0


    def calculate_sigma8_and_der(self,
                         # cosmo_model = self.cosmo_model,
                         **params_values_dict):
        
        params_values = params_values_dict.copy()
        update_params_with_defaults(params_values, self.emulator_dict[self.cosmo_model]['default'])
        

        params_dict = {}
        for k,v in zip(params_values.keys(),params_values.values()):
            params_dict[k]=[v]

        if 'm_ncdm' in params_dict.keys():
            if isinstance(params_dict['m_ncdm'][0],str): 
                params_dict['m_ncdm'] =  [float(params_dict['m_ncdm'][0].split(',')[0])]

        if self.jax_mode:
            self.cp_predicted_der = self.cp_der_nn[self.cosmo_model].predict(params_dict)
        else:
            self.cp_predicted_der = self.cp_der_nn[self.cosmo_model].ten_to_predictions_np(params_dict)[0]
        self.sigma8 = self.cp_predicted_der[1]
        self.Neff = self.cp_predicted_der[4]
        return 0


    def calculate_sigma8_at_z(self,
                             # cosmo_model = self.cosmo_model,
                             **params_values_dict):
        params_values = params_values_dict.copy()
        update_params_with_defaults(params_values, self.emulator_dict[self.cosmo_model]['default'])


        params_dict = {}
        for k,v in zip(params_values.keys(),params_values.values()):
            params_dict[k]=[v]

        if 'm_ncdm' in params_dict.keys():
            if isinstance(params_dict['m_ncdm'][0],str): 
                params_dict['m_ncdm'] =  [float(params_dict['m_ncdm'][0].split(',')[0])]


        s8z  = self.cp_s8_nn[self.cosmo_model].predictions_np(params_dict)
        # print(self.s8z)
        self.s8z_interp = scipy.interpolate.interp1d(
                                                    self.linspace(0.,20.,5000),
                                                    s8z[0],
                                                    kind='linear',
                                                    axis=-1,
                                                    copy=True,
                                                    bounds_error=None,
                                                    fill_value=np.nan,
                                                    assume_sorted=False)

    def calculate_pknl(self,
                      # cosmo_model = self.cosmo_model,
                      **params_values_dict):
        
        z_arr = self.cszfast_pk_grid_z


        k_arr = self.cszfast_pk_grid_k 


        params_values = params_values_dict.copy()
        update_params_with_defaults(params_values, self.emulator_dict[self.cosmo_model]['default'])


        params_dict = {}
        for k,v in zip(params_values.keys(),params_values.values()):
            params_dict[k]=[v]

        if 'm_ncdm' in params_dict.keys():
            if isinstance(params_dict['m_ncdm'][0],str): 
                params_dict['m_ncdm'] =  [float(params_dict['m_ncdm'][0].split(',')[0])]


        predicted_pk_spectrum_z = []

        for zp in z_arr:
            params_dict_pp = params_dict.copy()
            params_dict_pp['z_pk_save_nonclass'] = [zp]
            predicted_pk_spectrum_z.append(self.cp_pknl_nn[self.cosmo_model].predictions_np(params_dict_pp)[0])

        predicted_pk_spectrum = self.asarray(predicted_pk_spectrum_z)


        pk = 10.**predicted_pk_spectrum

        pk_re = pk*self.pk_power_fac
        pk_re = self.transpose(pk_re)


        self.pknl_interp = PowerSpectrumInterpolator(z_arr,k_arr,self.log(pk_re).T,logP=True)


        self.cszfast_pk_grid_pknl = pk_re
        self.cszfast_pk_grid_pknl_flat = pk_re.flatten()

        return pk_re, k_arr, z_arr
    



    def calculate_pkl_at_z(self,
                           z_asked,
                          params_values_dict=None):


        k_arr = self.cszfast_pk_grid_k 

        if params_values_dict:
            params_values = params_values_dict.copy()
        else:
            params_values = self.params_for_emulators

        update_params_with_defaults(params_values, self.emulator_dict[self.cosmo_model]['default'])


        params_dict = {}
        for k,v in zip(params_values.keys(),params_values.values()):
            params_dict[k]=[v]

        if 'm_ncdm' in params_dict.keys():
            if isinstance(params_dict['m_ncdm'][0],str): 
                params_dict['m_ncdm'] =  [float(params_dict['m_ncdm'][0].split(',')[0])]


        predicted_pk_spectrum_z = []

        params_dict_pp = params_dict.copy()
        update_params_with_defaults(params_dict_pp, self.emulator_dict[self.cosmo_model]['default'])

        params_dict_pp['z_pk_save_nonclass'] = [z_asked]
        if self.jax_mode:
            predicted_pk_spectrum_z.append(self.cp_pkl_nn[self.cosmo_model].predict(params_dict_pp))
        else:
            predicted_pk_spectrum_z.append(self.cp_pkl_nn[self.cosmo_model].predictions_np(params_dict_pp)[0])

        predicted_pk_spectrum = self.asarray(predicted_pk_spectrum_z)


        pk = 10.**predicted_pk_spectrum
        pk_re = pk*self.pk_power_fac
        pk_re = self.transpose(pk_re)

        
        return pk_re, k_arr
    

    def calculate_pknl_at_z(self,
                           z_asked,
                          params_values_dict=None):

        z_arr = self.cszfast_pk_grid_z

        k_arr = self.cszfast_pk_grid_k 

        if params_values_dict:

            params_values = params_values_dict.copy()

        else:

            params_values = self.params_for_emulators

        update_params_with_defaults(params_values, self.emulator_dict[self.cosmo_model]['default'])


        params_dict = {}
        for k,v in zip(params_values.keys(),params_values.values()):
            params_dict[k]=[v]

        if 'm_ncdm' in params_dict.keys():
            if isinstance(params_dict['m_ncdm'][0],str): 
                params_dict['m_ncdm'] =  [float(params_dict['m_ncdm'][0].split(',')[0])]


        predicted_pk_spectrum_z = []

        z_asked = z_asked
        params_dict_pp = params_dict.copy()
        update_params_with_defaults(params_dict_pp, self.emulator_dict[self.cosmo_model]['default'])

        params_dict_pp['z_pk_save_nonclass'] = [z_asked]
        predicted_pk_spectrum_z.append(self.cp_pknl_nn[self.cosmo_model].predictions_np(params_dict_pp)[0])

        predicted_pk_spectrum = self.asarray(predicted_pk_spectrum_z)


        pk = 10.**predicted_pk_spectrum
        pk_re = pk*self.pk_power_fac
        pk_re = self.transpose(pk_re)

        
        return pk_re, k_arr
    

    def calculate_hubble(self,
                         **params_values_dict):
        
        params_values = params_values_dict.copy()

        update_params_with_defaults(params_values, self.emulator_dict[self.cosmo_model]['default'])

        params_dict = {}
        for k,v in zip(params_values.keys(),params_values.values()):
            params_dict[k]=[v]

        if 'm_ncdm' in params_dict.keys():
            if isinstance(params_dict['m_ncdm'][0],str): 
                params_dict['m_ncdm'] =  [float(params_dict['m_ncdm'][0].split(',')[0])]


        if self.jax_mode:
            # print("JAX MODE in hubble")
            # self.cp_predicted_hubble = self.cp_h_nn[self.cosmo_model].ten_to_predictions_np(params_dict)[0]
            # print(params_dict)
            # print("params_dict type:", type(params_dict))
            self.cp_predicted_hubble = self.cp_h_nn[self.cosmo_model].predict(params_dict)
            # print("self.cp_predicted_hubble type:", type(self.cp_predicted_hubble))
            # print("self.cp_predicted_hubble",self.cp_predicted_hubble)

            
            # Assuming `cp_z_interp` and `cp_predicted_hubble` are JAX arrays
            def hz_interp(x):
                return jnp.interp(x, self.cp_z_interp, self.cp_predicted_hubble, left=jnp.nan, right=jnp.nan)

            self.hz_interp = hz_interp
            # exit()
        else:
            self.cp_predicted_hubble = self.cp_h_nn[self.cosmo_model].ten_to_predictions_np(params_dict)[0]
            # print("self.cp_predicted_hubble",self.cp_predicted_hubble)
            

            self.hz_interp = scipy.interpolate.interp1d(
                                                self.cp_z_interp,
                                                self.cp_predicted_hubble,
                                                kind='linear',
                                                axis=-1,
                                                copy=True,
                                                bounds_error=None,
                                                fill_value=np.nan,
                                                assume_sorted=False)

    def calculate_chi(self,
                      **params_values_dict):

        params_values = params_values_dict.copy()

        update_params_with_defaults(params_values, self.emulator_dict[self.cosmo_model]['default'])

        params_dict = {}
        for k,v in zip(params_values.keys(),params_values.values()):
            params_dict[k]=[v]

        if 'm_ncdm' in params_dict.keys():
            if isinstance(params_dict['m_ncdm'][0],str): 
                params_dict['m_ncdm'] =  [float(params_dict['m_ncdm'][0].split(',')[0])]

        if self.jax_mode: 
            # print("JAX MODE in chi")
            # self.cp_da_nn[self.cosmo_model].log = False
            self.cp_predicted_da = self.cp_da_nn[self.cosmo_model].predict(params_dict)
            # print("self.cp_predicted_da",self.cp_predicted_da)

            if self.cosmo_model == 'ede-v2':
                # print('ede-v2 case')
                self.cp_predicted_da = jnp.insert(self.cp_predicted_da, 0, 0)
            self.cp_predicted_da *= (1.+self.cp_z_interp)
    
            def chi_interp(x):
                return jnp.interp(x, self.cp_z_interp, self.cp_predicted_da, left=jnp.nan, right=jnp.nan)

            self.chi_interp = chi_interp

        else:
            # deal with different scaling of DA in different model from emulator training
            if self.cosmo_model == 'ede-v2':

                self.cp_predicted_da  = self.cp_da_nn[self.cosmo_model].ten_to_predictions_np(params_dict)[0]
                self.cp_predicted_da = np.insert(self.cp_predicted_da, 0, 0)
            
            else:
            
                self.cp_predicted_da  = self.cp_da_nn[self.cosmo_model].predictions_np(params_dict)[0]
            

            self.chi_interp = scipy.interpolate.interp1d(
                                                        self.cp_z_interp,
                                                        self.cp_predicted_da*(1.+self.cp_z_interp),
                                                        kind='linear',
                                                        axis=-1,
                                                        copy=True,
                                                        bounds_error=None,
                                                        fill_value=np.nan,
                                                        assume_sorted=False)

    def get_cmb_cls(self,ell_factor=True,Tcmb_uk = Tcmb_uk):

        cls = {}
        cls['ell'] = self.arange(self.cszfast_ldim)
        cls['tt'] = self.zeros(self.cszfast_ldim)
        cls['te'] = self.zeros(self.cszfast_ldim)
        cls['ee'] = self.zeros(self.cszfast_ldim)
        cls['pp'] = self.zeros(self.cszfast_ldim)
        cls['bb'] = self.zeros(self.cszfast_ldim)
        cls['tt'][2:self.cp_lmax+1] = (Tcmb_uk)**2.*self.cp_predicted_tt_spectrum.copy()
        cls['te'][2:self.cp_lmax+1] = (Tcmb_uk)**2.*self.cp_predicted_te_spectrum.copy()
        cls['ee'][2:self.cp_lmax+1] = (Tcmb_uk)**2.*self.cp_predicted_ee_spectrum.copy()
        cls['pp'][2:self.cp_lmax+1] = self.cp_predicted_pp_spectrum.copy()/4. ## this is clkk... works for so lensinglite lkl


        if ell_factor==False:
            fac_l = self.zeros(self.cszfast_ldim)
            fac_l[2:self.cp_lmax+1] = 1./(cls['ell'][2:self.cp_lmax+1]*(cls['ell'][2:self.cp_lmax+1]+1.)/2./self.pi)
            cls['tt'][2:self.cp_lmax+1] *= fac_l[2:self.cp_lmax+1]
            cls['te'][2:self.cp_lmax+1] *= fac_l[2:self.cp_lmax+1]
            cls['ee'][2:self.cp_lmax+1] *= fac_l[2:self.cp_lmax+1]
            # cls['bb'] = self.zeros(self.cszfast_ldim)
        return cls


    def get_pknl_at_k_and_z(self,k_asked,z_asked):
    # def get_pkl_at_k_and_z(self,k_asked,z_asked,method = 'cloughtocher'):
        # if method == 'linear':
        #     pk = self.pkl_linearnd_interp(z_asked,self.log(k_asked))
        # elif method == 'cloughtocher':
        #     pk = self.pkl_cloughtocher_interp(z_asked,self.log(k_asked))
        # return np.exp(pk)
        return self.pknl_interp.P(z_asked,k_asked)


    # def get_pkl_at_k_and_z(self,k_asked,z_asked,method = 'cloughtocher'):
    def get_pkl_at_k_and_z(self,k_asked,z_asked):
        # if method == 'linear':
        #     pk = self.pknl_linearnd_interp(z_asked,self.log(k_asked))
        # elif method == 'cloughtocher':
        #     pk = self.pknl_cloughtocher_interp(z_asked,self.log(k_asked))
        # return np.exp(pk)
        return self.pkl_interp.P(z_asked,k_asked)

    # function used to overwrite the classy function in fast mode.
    def get_sigma_at_r_and_z(self,r_asked,z_asked):
        # if method == 'linear':
        #     pk = self.pknl_linearnd_interp(z_asked,self.log(k_asked))
        # elif method == 'cloughtocher':
        #     pk = self.pknl_cloughtocher_interp(z_asked,self.log(k_asked))
        # return np.exp(pk)
        k = self.cszfast_pk_grid_k
        P_at_z = self.get_pkl_at_k_and_z(k,z_asked)
        R,var = TophatVar(k, lowring=True)(P_at_z, extrap=True)
        varR = CubicSpline(R, var)
        sigma_at_r_and_z = np.sqrt(varR(r_asked))
        return sigma_at_r_and_z



    def get_hubble(self, z,units="1/Mpc"):
        if self.jax_mode:
            return jnp.array(self.hz_interp(z)*H_units_conv_factor[units])
        else:
            return np.array(self.hz_interp(z)*H_units_conv_factor[units])

    def get_chi(self, z):
        if self.jax_mode:
            return jnp.array(self.chi_interp(z))
        else:
            return np.array(self.chi_interp(z))

    def get_gas_pressure_profile_x(self,z,m,x):
        return 0#np.vectorize(self.csz_base.get_pressure_P_over_P_delta_at_x_M_z_b12_200c)(x,m,z)


    # def get_gas_pressure_profile_x_parallel(self,index_z,param_values_array,**kwargs):
    #     # zp = self.cszfast_zgrid[iz]
    #     # x = self.get_gas_pressure_profile_x(zp,3e13,self.cszfast_gas_pressure_xgrid)
    #     x = 1
    #     return x


    def tabulate_gas_pressure_profile_k(self):
        z_asked,m_asked,x_asked = 0.2,3e14,self.geomspace(1e-3,1e2,500)
        start = time.time()
        px = self.get_gas_pressure_profile_x(z_asked,m_asked,x_asked)
        end = time.time()
        # print(px)
        # print('end tabuulate pressure profile:',end-start)
        # print('grid tabulate pressure profile')
        start = time.time()
        # px = self.get_gas_pressure_profile_x(self.cszfast_zgrid,self.cszfast_mgrid,self.cszfast_gas_pressure_xgrid)
        # for zp in self.cszfast_zgrid:
        #     for mp in self.cszfast_mgrid:
        #         zp = mp
        #         # px = self.get_gas_pressure_profile_x(zp,mp,self.cszfast_gas_pressure_xgrid)
        # zp,mp = 0.1,2e14
        px = self.get_gas_pressure_profile_x(self.cszfast_zgrid[:,None,None],
                                             self.cszfast_mgrid[None,:,None],
                                             self.cszfast_gas_pressure_xgrid[None,None,:])

        # fn=functools.partial(get_gas_pressure_profile_x_parallel,
        #                      param_values_array=None)
        # pool = multiprocessing.Pool()
        # results = pool.map(fn,range(self.cszfast_zgrid_nz))
        # pool.close()
        #
        # def task(iz):
        #     zp = self.cszfast_zgrid[iz]
        #     x = self.get_gas_pressure_profile_x(zp,m_asked,self.cszfast_gas_pressure_xgrid)
        # processes = [Process(target=task, args=(i,)) for i in range(self.cszfast_zgrid_nz)]
        #         # start all processes
        # for process in processes:
        #     process.start()
        # # wait for all processes to complete
        # for process in processes:
        #     process.join()
        # # report that all tasks are completed
        # print('Done', flush=True)
        end = time.time()

        # print(px)
        # print('end grid tabuulate pressure profile:',end-start)
    #
    # def get_gas_pressure_profile_x_parallel(index_z,param_values_array,**kwargs):
    #     # zp = self.cszfast_zgrid[iz]
    #     # x = self.get_gas_pressure_profile_x(zp,3e13,self.cszfast_gas_pressure_xgrid)
    #     x = 1
    #     return x


    def get_sigma8_at_z(self,z):
        return self.s8z_interp(z)


    def rs_drag(self):
        try:
            return self.cp_predicted_der[13]
        except AttributeError:
            return 0
    #################################
    # gives an estimation of f(z)*sigma8(z) at the scale of 8 h/Mpc, computed as (d sigma8/d ln a)
    # Now used by cobaya wrapper.
    def get_effective_f_sigma8(self, z, z_step=0.1,params_values_dict={}):
        """
        effective_f_sigma8(z)

        Returns the time derivative of sigma8(z) computed as (d sigma8/d ln a)

        Parameters
        ----------
        z : float
                Desired redshift
        z_step : float
                Default step used for the numerical two-sided derivative. For z < z_step the step is reduced progressively down to z_step/10 while sticking to a double-sided derivative. For z< z_step/10 a single-sided derivative is used instead.

        Returns
        -------
        (d ln sigma8/d ln a)(z) (dimensionless)
        """

        s8z_interp =  self.s8z_interp
        # we need d sigma8/d ln a = - (d sigma8/dz)*(1+z)

        # if possible, use two-sided derivative with default value of z_step
        if z >= z_step:
            result = (s8z_interp(z-z_step)-s8z_interp(z+z_step))/(2.*z_step)*(1+z)
            # return (s8z_interp(z-z_step)-s8z_interp(z+z_step))/(2.*z_step)*(1+z)
        else:
            # if z is between z_step/10 and z_step, reduce z_step to z, and then stick to two-sided derivative
            if (z > z_step/10.):
                z_step = z
                result = (s8z_interp(z-z_step)-s8z_interp(z+z_step))/(2.*z_step)*(1+z)
                # return (s8z_interp(z-z_step)-s8z_interp(z+z_step))/(2.*z_step)*(1+z)
            # if z is between 0 and z_step/10, use single-sided derivative with z_step/10
            else:
                z_step /=10
                result = (s8z_interp(z)-s8z_interp(z+z_step))/z_step*(1+z)
                # return (s8z_interp(z)-s8z_interp(z+z_step))/z_step*(1+z)
        # print('fsigma8 result : ',result)
        return result
