from cobaya.theories.classy import classy
from copy import deepcopy
from typing import NamedTuple, Sequence, Union, Optional
from cobaya.tools import load_module
import logging
import os
import numpy as np
import time

import warnings


class classy_sz(classy):

    use_class_sz_fast_mode = 1 # this is passed in the yaml file
    use_class_sz_no_cosmo_mode = 0 # this is passed in the yaml file
    lensing_lkl = 'ACT'

    baos = None 

    def initialize(self):
        """Importing CLASS from the correct path, if given, and if not, globally."""

        self.classy_module = self.is_installed()
        if not self.classy_module:
            raise NotInstalledError(
                self.log, "Could not find CLASS_SZ. Check error message above.")

        # Ignore the specific UserWarning from mcfit about using backend='jax'
        warnings.filterwarnings(
            "ignore",
            message="use backend='jax' if desired",
            category=UserWarning
        )

        from classy_sz import Class, CosmoSevereError, CosmoComputationError

        # import classy_sz
        # import os
        # path_to_data = os.getenv("PATH_TO_CLASS_SZ_DATA")
        # print(path_to_data)

        # print(classy_sz.__file__)

        # import classy_szfast
        # print(classy_szfast.__file__)
        # import sys; sys.exit()
        # print(sys.path)

        
        global CosmoComputationError, CosmoSevereError

        self.classy = Class()
        super(classy,self).initialize()


        # Add general CLASS stuff
        self.extra_args["output"] = self.extra_args.get("output", "")

        if "sBBN file" in self.extra_args:
            self.extra_args["sBBN file"] = (
                self.extra_args["sBBN file"].format(classy=self.path))
            
        # Derived parameters that may not have been requested, but will be necessary later
        self.derived_extra = []
        self.log.info("Initialized!")


        ## rename some parameters to avoid conflices
        classy_sz_renames = {

            'omega_m':'Omega_m',
            'Omegam':'Omega_m',
            'Omega_m':'Omega_m'
        }
        self.renames.update(classy_sz_renames)


        if self.use_class_sz_no_cosmo_mode == 1:

            self.log.info("Initializing cosmology part!")

            initial_parameters = self.extra_args.copy()


            self.classy.set(initial_parameters)
            self.classy.compute_class_szfast(likelihood_mode=True)
            self.log.info("cosmology part initialized!")




    def must_provide(self, **requirements):



        if "Cl_sz" in requirements:
            # make sure cobaya still runs as it does for standard classy
            requirements.pop("Cl_sz")
            # specify the method to collect the new observable
            self.collectors["Cl_sz"] = Collector(
                    method="cl_sz", # name of the method in classy.pyx
                    args_names=[],
                    args=[])

        if "Cl_yxg" in requirements:
            # make sure cobaya still runs as it does for standard classy
            requirements.pop("Cl_yxg")
            # specify the method to collect the new observable
            self.collectors["Cl_yxg"] = Collector(
                    method="cl_yg", # name of the method in classy.pyx
                    args_names=[],
                    args=[])

        if "Cl_gxg" in requirements:
            # make sure cobaya still runs as it does for standard classy
            requirements.pop("Cl_gxg")
            # specify the method to collect the new observable
            self.collectors["Cl_gxg"] = Collector(
                    method="cl_gg", # name of the method in classy.pyx
                    args_names=[],
                    args=[])


        if "Cl_gxmu" in requirements:
            # make sure cobaya still runs as it does for standard classy
            requirements.pop("Cl_gxmu")
            # specify the method to collect the new observable
            self.collectors["Cl_gxmu"] = Collector(
                    method="cl_gm", # name of the method in classy.pyx
                    args_names=[],
                    args=[])


        if "Cl_muxmu" in requirements:
            # make sure cobaya still runs as it does for standard classy
            requirements.pop("Cl_muxmu")
            # specify the method to collect the new observable
            self.collectors["Cl_muxmu"] = Collector(
                    method="cl_mm", # name of the method in classy.pyx
                    args_names=[],
                    args=[])


        if "Cl_kxg" in requirements:
            # make sure cobaya still runs as it does for standard classy
            requirements.pop("Cl_kxg")
            # specify the method to collect the new observable
            self.collectors["Cl_kxg"] = Collector(
                    method="cl_kg", # name of the method in classy.pyx
                    args_names=[],
                    args=[])

        if "Cl_kxmu" in requirements:
        # make sure cobaya still runs as it does for standard classy
            requirements.pop("Cl_kxmu")
        # specify the method to collect the new observable
            self.collectors["Cl_kxmu"] = Collector(
                    method="cl_km", # name of the method in classy.pyx
                    args_names=[],
                    args=[])


        if "Cl_yxmu" in requirements:
            # make sure cobaya still runs as it does for standard classy
            requirements.pop("Cl_yxmu")
            # specify the method to collect the new observable
            self.collectors["Cl_yxmu"] = Collector(
                    method="cl_ym", # name of the method in classy.pyx
                    args_names=[],
                    args=[])
        if "Cl_kgxg" in requirements:
            # make sure cobaya still runs as it does for standard classy
            requirements.pop("Cl_kgxg")
            # specify the method to collect the new observable
            self.collectors["Cl_kgxg"] = Collector(
                    method="cl_ggamma", # name of the method in classy.pyx
                    args_names=[],
                    args=[])
        if "Cl_kgxmu" in requirements:
                # make sure cobaya still runs as it does for standard classy
                requirements.pop("Cl_kgxmu")
                # specify the method to collect the new observable
                self.collectors["Cl_kgxmu"] = Collector(
                        method="cl_kg_m", # name of the method in classy.pyx
                        args_names=[],
                        args=[])
        if "Cl_IAxg" in requirements:
            # make sure cobaya still runs as it does for standard classy
            requirements.pop("Cl_IAxg")
            # specify the method to collect the new observable
            self.collectors["Cl_IAxg"] = Collector(
                    method="cl_IA_g", # name of the method in classy.pyx
                    args_names=[],
                    args=[])

        if "sz_binned_cluster_counts" in requirements:
            # make sure cobaya still runs as it does for standard classy
            requirements.pop("sz_binned_cluster_counts")
            # specify the method to collect the new observable
            self.collectors["sz_binned_cluster_counts"] = Collector(
                    method="dndzdy_theoretical", # name of the method in classy.pyx
                    args_names=[],
                    args=[])

        if "sz_unbinned_cluster_counts" in requirements:
            # make sure cobaya still runs as it does for standard classy
            requirements.pop("sz_unbinned_cluster_counts")
            # specify the method to collect the new observable
            self.collectors["sz_unbinned_cluster_counts"] = Collector(
                    method="szcounts_ntot_rates_loglike", # name of the method in classy.pyx
                    args_names=[],
                    args=[])

        if "cl_cib_kappa" in requirements:
                # make sure cobaya still runs as it does for standard classy
                requirements.pop("cl_cib_kappa")
                # specify the method to collect the new observable
                self.collectors["cl_cib_kappa"] = Collector(
                        method="cl_lens_cib", # name of the method in classy.pyx
                        args_names=[],
                        args=[])
        if "cl_galn_galn" in requirements:
                # make sure cobaya still runs as it does for standard classy
                requirements.pop("cl_galn_galn")
                # specify the method to collect the new observable
                self.collectors["cl_galn_galn"] = Collector(
                        method="cl_galn_galn", # name of the method in classy.pyx
                        args_names=[],
                        args=[])
        if "cl_galn_lens" in requirements:
                # make sure cobaya still runs as it does for standard classy
                requirements.pop("cl_galn_lens")
                # specify the method to collect the new observable
                self.collectors["cl_galn_lens"] = Collector(
                        method="cl_galn_lens", # name of the method in classy.pyx
                        args_names=[],
                        args=[])

        if "Cl_galnxtsz" in requirements:
                # make sure cobaya still runs as it does for standard classy
                requirements.pop("Cl_galnxtsz")
                # specify the method to collect the new observable
                self.collectors["Cl_galnxtsz"] = Collector(
                        method="cl_galn_tsz", # name of the method in classy.pyx
                        args_names=[],
                        args=[])
        if "Cl_galnxgallens" in requirements:
                # make sure cobaya still runs as it does for standard classy
                requirements.pop("Cl_galnxgallens")
                # specify the method to collect the new observable
                self.collectors["Cl_galnxgallens"] = Collector(
                        method="cl_galn_gallens", # name of the method in classy.pyx
                        args_names=[],
                        args=[])
        if "Cl_lensmagnxtsz" in requirements:
                # make sure cobaya still runs as it does for standard classy
                requirements.pop("Cl_lensmagnxtsz")
                # specify the method to collect the new observable
                self.collectors["Cl_lensmagnxtsz"] = Collector(
                        method="cl_lensmagn_tsz", # name of the method in classy.pyx
                        args_names=[],
                        args=[])
        if "Cl_lensmagnxgallens" in requirements:
                # make sure cobaya still runs as it does for standard classy
                requirements.pop("Cl_lensmagnxgallens")
                # specify the method to collect the new observable
                self.collectors["Cl_lensmagnxgallens"] = Collector(
                        method="cl_lensmagn_gallens", # name of the method in classy.pyx
                        args_names=[],
                        args=[])
        if "Cl_galnxIA" in requirements:
                # make sure cobaya still runs as it does for standard classy
                requirements.pop("Cl_galnxIA")
                # specify the method to collect the new observable
                self.collectors["Cl_galnxIA"] = Collector(
                        method="cl_galn_IA", # name of the method in classy.pyx
                        args_names=[],
                        args=[])

        ## added lensing fwith limber integral
        if "cl_lens_lens" in requirements:
                # make sure cobaya still runs as it does for standard classy
                requirements.pop("cl_lens_lens")
                # specify the method to collect the new observable
                self.collectors["cl_lens_lens"] = Collector(
                        method="cl_kk", # name of the method in classy.pyx
                        args_names=[],
                        args=[])
            
        super().must_provide(**requirements)


    # get the required new observable
    def get_Cl(self, ell_factor=False, units="FIRASmuK2"):
        
        if self.use_class_sz_fast_mode:

            return self.get_Clfast(ell_factor=ell_factor)
        
        else:
        
            return self._get_Cl(ell_factor=ell_factor, units=units, lensed=True)

    def get_Clfast(self,ell_factor = False):

        cls = {}
        cls = deepcopy(self._current_state["Cl"])

        lcp = np.asarray(cls['ell'])

        if ell_factor==True:
            cls['tt'] *= (2.7255e6)**2.*(lcp*(lcp+1.))/2./np.pi
            cls['te'] *= (2.7255e6)**2.*(lcp*(lcp+1.))/2./np.pi
            cls['ee'] *= (2.7255e6)**2.*(lcp*(lcp+1.))/2./np.pi

        if self.lensing_lkl == "ACT" and ell_factor == False:
            cls['tt'] *= (2.7255e6)**2.
            cls['te'] *= (2.7255e6)**2.
            cls['ee'] *= (2.7255e6)**2.

        if self.lensing_lkl ==  "SOLikeT":
            cls['pp'] *= (lcp*(lcp+1.))**2./4.

        elif self.lensing_lkl == "ACT":
            cls['pp'] *= 1.
             
        else: # here for the planck lensing lkl, using lfactor option gives:
            cls['pp'] *= (lcp*(lcp+1.))**2.*1./2./np.pi

        return cls
    
    # get the required new observable
    def get_Cl_sz(self):
        cls = {}
        cls = deepcopy(self._current_state["Cl_sz"])
        return cls

    def get_Cl_yxg(self):
        cls = {}
        cls = deepcopy(self._current_state["Cl_yxg"])
        return cls
    def get_Cl_kxg(self):
        cls = {}
        cls = deepcopy(self._current_state["Cl_kxg"])
        return cls
    def get_Cl_gxg(self):
        cls = {}
        cls = deepcopy(self._current_state["Cl_gxg"])
        return cls
    def get_Cl_muxmu(self):
        cls = {}
        cls = deepcopy(self._current_state["Cl_muxmu"])
        return cls
    def get_Cl_gxmu(self):
        cls = {}
        cls = deepcopy(self._current_state["Cl_gxmu"])
        return cls
    def get_Cl_kxmu(self):
        cls = {}
        cls = deepcopy(self._current_state["Cl_kxmu"])
        return cls
    def get_Cl_yxmu(self):
        cls = {}
        cls = deepcopy(self._current_state["Cl_yxmu"])
        return cls
    def get_Cl_kgxg(self):
        cls = {}
        cls = deepcopy(self._current_state["Cl_kgxg"])
        return cls
    def get_Cl_kgxmu(self):
        cls = {}
        cls = deepcopy(self._current_state["Cl_kgxmu"])
        return cls
    def get_Cl_IAxg(self):
        cls = {}
        cls = deepcopy(self._current_state["Cl_IAxg"])
        return cls
    def get_cl_galn_lens(self):
        cls = {}
        cls = deepcopy(self._current_state["cl_galn_lens"])
        return cls
    def get_cl_galn_galn(self):
        cls = {}
        cls = deepcopy(self._current_state["cl_galn_galn"])
        return cls
    def get_Cl_galnxtsz(self):
        cls = {}
        cls = deepcopy(self._current_state["Cl_galnxtsz"])
        return cls
    def get_Cl_galnxgallens(self):
        cls = {}
        cls = deepcopy(self._current_state["Cl_galnxgallens"])
        return cls
    def get_Cl_lensmagnxtsz(self):
        cls = {}
        cls = deepcopy(self._current_state["Cl_lensmagnxtsz"])
        return cls
    def get_Cl_lensmagnxgallens(self):
        cls = {}
        cls = deepcopy(self._current_state["Cl_lensmagnxgallens"])
        return cls
    def get_Cl_galnxIA(self):
        cls = {}
        cls = deepcopy(self._current_state["Cl_galnxIA"])
        return cls

    # get the required new observable
    def get_sz_unbinned_cluster_counts(self):

        cls = deepcopy(self._current_state["sz_unbinned_cluster_counts"])

        return cls['loglike'],cls['ntot'],cls['rates']


    # get the required new observable
    def get_sz_binned_cluster_counts(self):
        cls = {}
        cls = deepcopy(self._current_state["sz_binned_cluster_counts"])
        return cls

    def get_cl_cib_kappa(self):
        cls = {}
        cls = deepcopy(self._current_state["cl_cib_kappa"])
        return cls

    def get_cl_lens_lens(self):
            cls = {}
            cls = deepcopy(self._current_state["cl_lens_lens"])
            return cls

    # IMPORTANT: this method is imported from cobaya and modified to accomodate the emulators
    def calculate(self, state, want_derived=True, **params_values_dict):

        
        params_values = params_values_dict.copy()
        # if baos are requested we need to update the relevant flags
        if self.baos:
            params_values.update({'skip_chi':0,'skip_hubble':0})

        if 'N_ncdm' in self.extra_args.keys():

            if self.extra_args['N_ncdm'] ==  3:
            
                str_mncdm = str(params_values['m_ncdm'])
                params_values['m_ncdm'] = str_mncdm+','+str_mncdm+','+str_mncdm

        try:
            
            params_values['ln10^{10}A_s'] = params_values.pop("logA")
            
            self.set(params_values)
        
        except KeyError:
        
            self.set(params_values)

        # Compute!
        try:
            if self.use_class_sz_fast_mode == 1:


                if self.use_class_sz_no_cosmo_mode == 1:

                    start_time = time.time()
                    self.classy.compute_class_sz(params_values)
                    end_time = time.time()
                    # self.log.info("Execution time of class_sz: {:.5f} seconds".format(end_time - start_time))

                else:

                    start_time = time.time()
                    # self.classy = Class()
                    # print('pars in classy',self.classy.pars)
                    self.classy.compute_class_szfast(likelihood_mode=True)
                    end_time = time.time()
                    # self.log.info("Execution time of class_szfast: {:.5f} seconds".format(end_time - start_time))
                    
                    # some debug printouts
                    # cl_kk_hm = self.classy.cl_kk
                    # ell = np.asarray(cl_kk_hm()['ell'])
                    # fac = ell*(ell+1.)/2./np.pi
                    # cl_kk_hf = np.asarray(cl_kk_hm()['hf'])/fac
                    # print('cl_kk_hf',cl_kk_hf[0])
                    # print('ell',ell[0])
                    # print('pp',self.classy.lensed_cl()['pp'][2])
                    # print('angular_distance',self.classy.angular_distance(0.1))
                    # print('Omega_Lambda',self.classy.Omega_Lambda())
                    # print('Hubble',self.classy.Hubble(0))
                    # print('Omega_m',self.classy.Omega_m())
                    # print('Neff',self.classy.Neff())
                    # print('m_ncdm_tot',self.classy.get_current_derived_parameters(['m_ncdm_tot']))
                    # print('Neff',self.classy.get_current_derived_parameters(['Neff']))
                    # print('Omega_m',self.classy.get_current_derived_parameters(['Omega_m']))
                    # print('h',self.classy.get_current_derived_parameters(['h']))
                    # print('m_ncdm_in_eV',self.classy.get_current_derived_parameters(['m_ncdm_in_eV']))
                    # print("\n\n-----------------------------------\n\n")
                    # import sys; sys.exit()
            else:

                self.classy.compute()

        # "Valid" failure of CLASS: parameters too extreme -> log and report
        except self.classy_module.CosmoComputationError as e:
            if self.stop_at_error:
                self.log.error(
                    "Computation error (see traceback below)! "
                    "Parameters sent to CLASS: %r and %r.\n"
                    "To ignore this kind of error, make 'stop_at_error: False'.",
                    state["params"], dict(self.extra_args))
                raise
            else:
                self.log.debug("Computation of cosmological products failed. "
                               "Assigning 0 likelihood and going on. "
                               "The output of the CLASS error was %s" % e)
            return False
        # CLASS not correctly initialized, or input parameters not correct
        except self.classy_module.CosmoSevereError:
            self.log.error("Serious error setting parameters or computing results. "
                           "The parameters passed were %r and %r. To see the original "
                           "CLASS' error traceback, make 'debug: True'.",
                           state["params"], self.extra_args)
            raise  # No LoggedError, so that CLASS traceback gets printed
        # Gather products
        for product, collector in self.collectors.items():
            # print(product,collector)
            # Special case: sigma8 needs H0, which cannot be known beforehand:
            # if "sigma8" in self.collectors:
            #     self.collectors["sigma8"].args[0] = 8 / self.classy.h()
            method = getattr(self.classy, collector.method)
            arg_array = self.collectors[product].arg_array
            if isinstance(arg_array, int):
                arg_array = np.atleast_1d(arg_array)
            if arg_array is None:
                state[product] = method(
                    *self.collectors[product].args, **self.collectors[product].kwargs)
            elif isinstance(arg_array, Sequence) or isinstance(arg_array, np.ndarray):
                arg_array = np.array(arg_array)
                if len(arg_array.shape) == 1:
                    # if more than one vectorised arg, assume all vectorised in parallel
                    n_values = len(self.collectors[product].args[arg_array[0]])
                    state[product] = np.zeros(n_values)
                    args = deepcopy(list(self.collectors[product].args))
                    for i in range(n_values):
                        for arg_arr_index in arg_array:
                            args[arg_arr_index] = \
                                self.collectors[product].args[arg_arr_index][i]
                        state[product][i] = method(
                            *args, **self.collectors[product].kwargs)
                elif len(arg_array.shape) == 2:
                    if len(arg_array) > 2:
                        raise NotImplementedError("Only 2 array expanded vars so far.")
                    # Create outer combinations
                    x_and_y = np.array(np.meshgrid(
                        self.collectors[product].args[arg_array[0, 0]],
                        self.collectors[product].args[arg_array[1, 0]])).T
                    args = deepcopy(list(self.collectors[product].args))
                    result = np.empty(shape=x_and_y.shape[:2])
                    for i, row in enumerate(x_and_y):
                        for j, column_element in enumerate(x_and_y[i]):
                            args[arg_array[0, 0]] = column_element[0]
                            args[arg_array[1, 0]] = column_element[1]
                            result[i, j] = method(
                                *args, **self.collectors[product].kwargs)
                    state[product] = (
                        self.collectors[product].args[arg_array[0, 0]],
                        self.collectors[product].args[arg_array[1, 0]], result)
                else:
                    raise ValueError("arg_array not correctly formatted.")
            elif arg_array in self.collectors[product].kwargs:
                value = np.atleast_1d(self.collectors[product].kwargs[arg_array])
                state[product] = np.zeros(value.shape)
                for i, v in enumerate(value):
                    kwargs = deepcopy(self.collectors[product].kwargs)
                    kwargs[arg_array] = v
                    state[product][i] = method(
                        *self.collectors[product].args, **kwargs)
            else:
                raise LoggedError(self.log, "Variable over which to do an array call "
                                            f"not known: arg_array={arg_array}")
            if collector.post:
                state[product] = collector.post(*state[product])
        # Prepare derived parameters

        d, d_extra = self._get_derived_all(derived_requested=want_derived)

        if want_derived:

            state["derived"] = {p: d.get(p) for p in self.output_params}


        state["derived_extra"] = deepcopy(d_extra)


    def set(self, params_values_dict):
        # If no output requested, remove arguments that produce an error
        # (e.g. complaints if halofit requested but no Cl's computed.) ?????
        # Needed for facilitating post-processing
        if not self.extra_args["output"]:
            for k in ["non_linear"]:
                self.extra_args.pop(k, None)
        # Prepare parameters to be passed: this-iteration + extra
        args = {self.translate_param(p): v for p, v in params_values_dict.items()}
        args.update(self.extra_args)
        # Generate and save
        # print("Setting parameters: %r", args)
        self.classy.set(**args)



    @classmethod
    def is_installed(cls, **kwargs):
        return load_module('classy_sz')


# this just need to be there as it's used to fill-in self.collectors in must_provide:
class Collector(NamedTuple):
    method: str
    args: Sequence = []
    args_names: Sequence = []
    kwargs: dict = {}
    arg_array: Union[int, Sequence] = None
    post: Optional[callable] = None
