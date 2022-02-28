'''

'''

import nibabel as nib
import os
from os import system as scmd
import numpy as np
import scipy.io as sio
import random

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn
from matplotlib.ticker import PercentFormatter


# %% Class
EPSILON = 1e-6
SEED = 372981654
class WMTI_RNN_Estimator:
    '''
    # DKI: 
    #   md, ad, rd, mk, ak, rk (mean/axial/radial diffusivity, mean/axial/radial kurtosis) maps,
    # WMTI (WM model parameter maps):
    #   f (axonal water fraction), Da (axonal diffusivity), Depar, Deperp(extra-axonal
    #   parallel and perpendicular diffusivities), c2 (mean cos ^ 2 of the axon
    #   orientation dispersion: c2 = 1 / 3 fully isotropic, c2 = 1 perfectly parallel)
    #   c2 is directly related to the Watson distribution concentration parameter

    # All diffusivities in um2 / ms
    # md, ad, rd should also be in um2 / ms, otherwise converted here
    # mask: brain or ROI mask

    '''
    def __init__(self, estimator_path=None, checkpoint=1000, batch_size=2048, 
                    wmti_bounds=[[0.01, 1], [0.01, 3.9], [0.01, 3], [0.01, 3], [0.33, 1]]):
        if estimator_path is None:
            root_path = os.path.dirname(os.path.realpath(__file__))
            self.model_path = os.path.join(root_path, 'model')
        else:
            self.model_path = estimator_path
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.filter_bounds = wmti_bounds
        self.init_attr()


    def init_attr(self, output_path=None):
        self.output = output_path
        self.dki_names = None
        self.wmti_names = None         
        self.tmp_path = None
        self.dki = None
        self.wmti = None
        self.header = None
        self.image_size = None
        self.sel_mask = None
        self.emb_position = None
        self.embedded_file = None
        self.estimate_file = None
        self.wmti_estimate = None
        self.filter = None
        if not (output_path is None or os.path.exists(self.output)): 
            os.makedirs(self.output)

    def __version__(self):
        print("1.0.0 (25.02.2022)")

    def read_dki_nii(self, dki_path, output_path, wmti_path=None, dki_names=['md', 'ad', 'rd', 'mk', 'ak', 'rk'], 
                wmti_names=['f', 'Da', 'Depar', 'Deperp', 'c2'], fa_threshold=0, fa_filename='fa.nii', mask=None):
 
        self.init_attr(output_path)
        self.dki_names = dki_names
        self.wmti_names = wmti_names 
        dki_list=[]
        fa_nii = nib.load(os.path.join(dki_path, fa_filename))
        self.header = fa_nii.header
        fa_map = fa_nii.get_fdata()
        self.image_size = fa_map.shape
        fa = fa_map.reshape(-1)

        for dn in self.dki_names:
            dki_vector = _get_image_from_nii(os.path.join(dki_path,f'{dn}.nii')).reshape(-1)
            dki_list.append(dki_vector)
        self.dki = np.array(dki_list).T     

        self.wmti = None
        if wmti_path is not None:
            wmti_list=[]   
            for wn in self.wmti_names:
                wmti_vector = _get_image_from_nii(os.path.join(wmti_path,f'{wn}.nii')).reshape(-1)
                wmti_list.append(wmti_vector)
            self.wmti = np.array(wmti_list).T 

        md,ad,rd,mk,ak,rk = self.dki.T
        if np.nanquantile(md[md>0],0.8) < 0.02:
            print("Multiply DKI by 1000")        
            md = md * 1e3
            ad = ad * 1e3
            rd = rd * 1e3
            self.dki[:,:3] = self.dki[:,:3] * 1e3

        dki_filter = np.array([fa>fa_threshold, (md>0)==(md<3), (ad>0)==(ad<3),(rd>0)==(rd<3),
                            (mk>0)==(mk<10), (ak>0)==(ak<10),(rk>0)==(rk<10)])
        if mask is not None: 
            roimask = _get_image_from_nii(mask).reshape(-1)
            dki_filter =  np.vstack((dki_filter, roimask>0))

        constraints = f'{fa_threshold} < fa & 0.0<md & md<3 & 0.0<ad & ad<3 & 0.0<rd & rd<3'\
                        '& 0.0<mk & mk<10 & 0.0<ak & ak<10 & 0.0<rk & rk<10'

        self.sel_mask = np.all(dki_filter, axis=0)
        self.dki = self.dki[self.sel_mask,:]
        if self.wmti is not None:
            self.wmti = self.wmti[self.sel_mask,:]
            self.wmti[np.isnan(self.wmti)] = 0
        
        print(f"DKI constraints: {constraints}")   

    def embed(self, embedded_filename='embedded_data', embedding_file=None, ratio=5, seed=SEED):
        if embedding_file is None:
            embedding_file = os.path.join(self.model_path, 'embedding_data.mat')
        self.tmp_path = os.path.join(self.output, "tmp")
        if not os.path.exists(self.tmp_path): os.makedirs(self.tmp_path)
        emb_mat = sio.loadmat(embedding_file)
        emb_dki = emb_mat['dki'].astype(np.float32)
        test_sz = self.dki.shape[0]
        n = min(test_sz * ratio, emb_dki.shape[0])
        total_sz = test_sz + n
        print(f"Test size: {test_sz}; Total_sz: {total_sz}")
        print(f"Actual embedding ratio: {round(n/test_sz)}")
        emb_dki = emb_dki[0:n,:]
        dki = np.full((total_sz, self.dki.shape[1]), np.nan)
        self.emb_position = np.zeros(total_sz)
        random.seed(seed)
        test_pos = np.array(sorted(random.sample(range(total_sz),test_sz)))
        self.emb_position[test_pos] = 1
        assert test_sz==np.sum(self.emb_position)
        dki[self.emb_position==1,:] = self.dki
        dki[self.emb_position==0,:] = emb_dki

        output_mat = {'emb_position': self.emb_position, 'dki':dki}
        if self.wmti is not None:
            emb_wmti = emb_mat['wmti_paras'].astype(np.float32)
            emb_wmti = emb_wmti[0:n,:]
            wmti_paras = np.full((total_sz, self.wmti.shape[1]), np.nan)
            wmti_paras[self.emb_position==1,:] = self.wmti
            wmti_paras[self.emb_position==0,:] = emb_wmti  
            output_mat['wmti_paras']  = wmti_paras

        self.embedded_file = os.path.join(self.tmp_path, f"{embedded_filename}.mat")
        sio.savemat(self.embedded_file, output_mat)    

    def test_(self, exec=True):
        assert self.wmti is not None, "Target WMTI is required for testing !"
        cmdline = f"{self.model_path}/main.py --mode=test --datapath={self.output} --dataset={self.embedded_file} " \
                f"--test_save_folder={self.tmp_path} --model_folder={self.model_path} "\
                f"--batch_size={self.batch_size} --train_perc=0 --val_perc=0 --load_checkpoint={self.checkpoint}"  
        self.estimate_file = os.path.join(self.tmp_path, 'test_pred_tgt.npy')                 
        if exec: 
            _execute_cmd('python3', cmdline)         

    def estimate_(self, exec=True):
        cmdline = f"{self.model_path}/wmti_estimate.py --datapath={self.output} --dataset={self.embedded_file} " \
                    f"--output_dir={self.tmp_path} --model_folder={self.model_path} "\
                    f"--batch_size={self.batch_size} --input_scale_type=3 --output_scale_type=2 --load_checkpoint={self.checkpoint}"  
        self.estimate_file = os.path.join(self.tmp_path, 'wmti_estimate.npy')                 
        if exec: 
            _execute_cmd('python3', cmdline)          

    def deembed(self):
        estimations = np.load(self.estimate_file)
        shapes = estimations.shape        
        pred_ = None
        if len(shapes)==2:
            pred_ = estimations
        elif len(shapes)==3:
            pred_ = estimations[:,:,0]
            if shapes[2]==2: 
                target_ = estimations[:,:,1]
                test_target_ = target_[self.emb_position==1,:]
                if self.wmti is not None:
                    delta = np.abs(self.wmti - test_target_).flatten()
                    assert np.max(delta)<EPSILON, 'Mismatching targets!'

        if pred_ is not None: 
            self.wmti_estimate = pred_[self.emb_position==1,:]

    def estimate(self, embedding_file=None, embedding_ratio=5, seed=SEED):
        self.embed(embedding_file=embedding_file, ratio=embedding_ratio, seed=seed)
        self.estimate_()
        self.deembed()

    def test(self, exec=True, filtering=False, embedding_file=None, embedding_ratio=5, seed=SEED):
        if self.wmti is None: return None

        self.embed(embedding_file=embedding_file, ratio=embedding_ratio, seed=seed)
        self.test_(exec=exec)
        self.deembed()
        self.evaluate_(filtering=filtering)

    def reconstruct_maps(self, save_path=None, filtering=False):

        assert self.wmti_estimate.shape[1]==len(self.wmti_names)
        if save_path is None: save_path = self.output

        self.filter_estimation(filtering=filtering)
        if self.filter is not None:
            self.sel_mask[self.sel_mask] = self.filter  

        for i, nm in enumerate(self.wmti_names):
            dl = np.full(self.sel_mask.shape, np.nan)               
            dl[self.sel_mask] = self.wmti_estimate[:, i] if self.filter is None else self.wmti_estimate[self.filter, i]
            dl_map = dl.reshape(self.image_size)
            
            newData = nib.Nifti1Image(dl_map, None, header=self.header)
            nib.save(newData, os.path.join(save_path, f"{nm}.nii"))  

        print(f"WMIT maps saved to {save_path}")

    def filter_estimation(self, filtering=True):      
        if filtering:
            filter_ = []
            wmti_constraint = ''
            for i, wn in enumerate(self.wmti_names):
                bd = self.filter_bounds[i]
                wm = self.wmti_estimate[:,i]
                filter_.append((bd[0]<wm) == (wm<bd[1]))
                wmti_constraint += f" {bd[0]}<{wn}<{bd[1]} &"

            wmti_constraint = wmti_constraint[:-1]
            filter = np.array(filter_)
            self.filter = np.all(filter, axis=0)
            print(f"filtering estimation: {wmti_constraint}")
        else:
            self.filter = None

    def evaluate_(self, filtering=False, tolerance=5., nbins=400, ymax=0.05, target_label="NLLS", estimate_label='RNN',
                plotparam = [r'$f$', r'$D_a$', r'$D_{e,\parallel}$', r'$D_{e,\perp}$', r'$c_2$']):
        if self.wmti is None: 
            return None

        self.filter_estimation(filtering=filtering)
        if self.filter is None:
            wmti = self.wmti
            wmti_estimate = self.wmti_estimate
        else:
            wmti = self.wmti[self.filter, :]
            wmti_estimate = self.wmti_estimate[self.filter, :]       

        plot_density_scatter(wmti, wmti_estimate, plotparam=plotparam, title=None, savetitle=f'{target_label}_vs_{estimate_label}', 
                                    savedir=self.tmp_path, xlabel=target_label, ylabel=estimate_label, show_colorbar=False, showfig=False)

        errors = np.abs((wmti_estimate - wmti))/(wmti+EPSILON) * 100 + EPSILON/100  # in %
        plot_error_distribution(errors, plotparam=plotparam, savedir=self.tmp_path, err_perc=tolerance, nbins=nbins, ymax=ymax,
                                    savetitle=f'{target_label}_{estimate_label}_diff_dist_{round(tolerance)}%', showfig=False) 

        print(f"Evaluation results saved to {self.tmp_path}")
                   
def _get_image_from_nii(nii_file):
    nii = nib.load(nii_file)
    return nii.get_fdata()

def _execute_cmd(cmd, pars_line, debug=False):
    cmd_line = cmd + " " + pars_line
    if debug: print(cmd_line)
    scmd(cmd_line) 

def density_scatter(x, y, ax=None, sort=True, bins=20, **kwargs):
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None:
        fig, ax = plt.subplots()

    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    z = interpn((0.5*(x_e[1:] + x_e[:-1]), 0.5*(y_e[1:]+y_e[:-1])),
                data, np.vstack([x, y]).T, method="splinef2d", bounds_error=False)
    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0
    # Sort the points by density, so that the densest points are plotted last
    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter(x, y, c=z, s=0.2, alpha=0.5, **kwargs)
    ax.set_aspect('equal')
    norm = Normalize(vmin=np.min(z), vmax=np.max(z))

    return ax, norm

def plot_density_scatter(target, pred, plotparam, plot_err=0.1, lim = [[0, 1], [0, 4], [0, 3], [0, 2], [0.3, 1]], xlabel='target', ylabel='prediction',
                         bins=[15, 15], title=None, savetitle=None, savedir=None, show_colorbar=False, showfig=True):
    fig, axes = plt.subplots(1, len(plotparam), dpi=150, figsize=(18, 4))
    plt.suptitle(title)
    for ii, par in enumerate(plotparam):
        ax = axes.flatten()[ii] if len(plotparam) > 1 else axes

        llim, hlim = lim[ii][0], lim[ii][1]
        x = np.linspace(llim, hlim)
        ax.plot(x, x, 'k', linewidth=0.5)
        ax.plot(x, x*(1+plot_err), 'k--', linewidth=0.5)
        ax.plot(x, x*(1-plot_err), 'k--', linewidth=0.5)
        ax, norm = density_scatter(
            target[:, ii], pred[:, ii], ax, sort=True, bins=bins)
        temp = (hlim-llim)/10
        ax.set_xlim(llim - temp, hlim +
                    temp), ax.set_ylim(llim - temp, hlim + temp)
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_title(par, fontsize=18)
        ax.yaxis.set_tick_params(labelsize=14)
        ax.xaxis.set_tick_params(labelsize=14)

    if show_colorbar:
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm),
                            ax=ax, shrink=0.6, ticks=range(10, 150, 20))
        cbar.ax.set_ylabel('Density')
    plt.tight_layout()
    if None not in [savedir, savetitle]:
        plt.savefig(f"{savedir}/{savetitle}.png")
    if showfig: plt.show()


def plot_error_distribution(errors, plotparam, savedir=None, err_perc=5, savetitle='Error_dist', ymax=0.05,
                            bins_range=(1e-4, 1e2), nbins=300, showfig=True):
    perc = err_perc  # %
    errors = np.round_(errors, 4)
    nsamples = errors.shape[0]
    print(f"nSamples: {nsamples}")

    fig, axes = plt.subplots(1, len(plotparam), figsize=(30, 6))
    for ii, par in enumerate(plotparam):
        ax = axes.flatten()[ii] if len(plotparam) > 1 else axes
        logbins = np.logspace(
            np.log10(bins_range[0]), np.log10(bins_range[-1]), nbins)
        logbins = np.unique(np.round(logbins, 3))
        hist_, _, _ = ax.hist(errors[:, ii], weights=np.ones(
            nsamples)/nsamples, bins=logbins)

        idx_less_than = np.sum(logbins <= perc)-1
        perc_ = logbins[idx_less_than]
        perc_lt = round(np.sum(hist_[0:idx_less_than]) * 100, 2)  # %,
        tex = f"Portion(err <= {perc_}%): {perc_lt}%"
        ax.set_xlim(1e-4, 1e2)
        ax.set_xscale('log')
        ax.set_xlabel(f'err (%) \n {tex}')
        ax.set_ylabel('Probability')
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylim(0, ymax)
        ax.set_aspect('auto')

    plt.tight_layout()
    if None not in [savedir, savetitle]:
        plt.savefig(f"{savedir}/{savetitle}.png")
    if showfig: plt.show()
