#/usr/bin/env python
# --
# quicklens/examples/lens/get_phi_map.py
# --
# generates a set of lensed maps in the flat-sky limit, then runs
# quadratic lensing estimators on them to estimate phi.
# Plots map of reconstructed phi.  EB.

import pdb
import numpy as np
import matplotlib.pyplot as pl
import pdb

import quicklens as ql

# simulation parameters.
nsims      = 25
lmin       = 100
lmax       = 3000
nx         = 256 # number of pixels.
dx         = 1./60./180.*np.pi # pixel width in radians.

nlev_t     = 5.0  # temperature noise level, in uK.arcmin.
nlev_p     = 5.0  # polarization noise level, in uK.arcmin.
bl         = ql.spec.bl(1., lmax) # beam transfer function.

pix        = ql.maps.pix(nx,dx)

mask       = np.ones( (nx, nx) ) # mask to apply when inverse-variance filtering.
                                 # currently, no masking.
                                 # alternatively, cosine masking:
                                 # x, y = np.meshgrid( np.arange(0,nx), np.arange(0,nx) )
                                 # mask = np.sin( np.pi/nx*x )*np.sin( np.pi/nx*y )

mc_sims_mf = None                # indices of simulations to use for estimating a mean-field.
                                 # currently, no mean-field subtraction.
                                 # alternatively: np.arange(nsims, 2*nsims)

npad       = 1

# plotting parameters.
t          = lambda l: (l+0.5)**4/(2.*np.pi) # scaling to apply to cl_phiphi when plotting.
lbins      = np.linspace(10, lmax, 20)       # multipole bins.

# cosmology parameters.
cl_unl     = ql.spec.get_camb_scalcl(lmax=lmax)
cl_len     = ql.spec.get_camb_lensedcl(lmax=lmax)
clpp       = ql.spec.cl2cfft(cl_unl.clpp, ql.maps.cfft(nx,dx)).get_ml(lbins, t=t)

# make libraries for simulated skies.
sky_lib    = ql.sims.cmb.library_flat_lensed(pix, cl_unl, "temp5/sky")
obs_lib    = ql.sims.obs.library_white_noise(pix, bl, sky_lib, nlev_t=nlev_t, nlev_p=nlev_p, lib_dir="temp5/obs")

# make libraries for inverse-variance filtered skies.
cl         = ql.spec.clmat_teb( ql.util.dictobj( {'lmax' : lmax, 'cltt' : cl_len.cltt, 'clee' : cl_len.clee, 'clbb' : cl_len.clbb} ) )
transf     = ql.spec.cl2tebfft(ql.util.dictobj( {'lmax' : lmax, 'cltt' : bl, 'clee' : bl, 'clbb' : bl} ), pix)
ivf_lib    = ql.sims.ivf.library_l_mask( ql.sims.ivf.library_diag(obs_lib, cl=cl, transf=transf, nlev_t=nlev_t, nlev_p=nlev_p, mask=mask), lmin=lmin, lmax=lmax )

qest_lib = ql.sims.qest.library(cl_unl, cl_len, ivf_lib, lib_dir="temp5/qest", npad=npad)
qest_lib_kappa = ql.sims.qest.library_kappa(qest_lib, sky_lib)

# --
# run estimators, make plots.
# --

#777 need to do mean field subtraction

tqu_len = sky_lib.get_sim_tqu(0)
phi_true = sky_lib.get_sim_phi(0)
kappa_true = sky_lib.get_sim_kappa(0)

estimator = 'ptt'
qft = qest_lib.get_sim_qft(estimator, 0)
qr = qest_lib.get_qr(estimator)
phihat_tt = qft / qr
phihat_tt.fft = np.nan_to_num(phihat_tt.fft)

test1 = phihat_tt.get_rffts()[0].get_rmap().map
#This is what code is doing: take inverse fft, take real part, take fft, take inverse fft
test3 = np.fft.irfft2(np.fft.rfft2(np.real(np.fft.ifft2(phihat_tt.fft)))*np.sqrt((phihat_tt.nx*phihat_tt.ny)/(phihat_tt.dx*phihat_tt.dy)))
#test 4: take inverse fft, take real part
test4 = np.real(np.fft.ifft2(phihat_tt.fft*np.sqrt((phihat_tt.nx*phihat_tt.ny)/(phihat_tt.dx*phihat_tt.dy))))
                     
estimator = 'pee'
qft = qest_lib.get_sim_qft(estimator, 0)
qr = qest_lib.get_qr(estimator)
phihat_ee = qft / qr
phihat_ee.fft = np.nan_to_num(phihat_ee.fft)

estimator = 'peb'
qft = qest_lib.get_sim_qft(estimator, 0)
qr = qest_lib.get_qr(estimator)
phihat_eb = qft / qr
phihat_eb.fft = np.nan_to_num(phihat_eb.fft)

#Plotting
#pl.rc('text', usetex=True)
#pl.rc('font', family='serif')
fig, axarr = pl.subplots(2,4)
fig.suptitle('CMB Lensing')

axarr[0,0].imshow(phi_true.map)
axarr[0,0].get_xaxis().set_visible(False)
axarr[0,0].get_yaxis().set_visible(False)
axarr[0,0].set_title('True Phi')

axarr[0,1].imshow(tqu_len.tmap)
axarr[0,1].get_xaxis().set_visible(False)
axarr[0,1].get_yaxis().set_visible(False)
axarr[0,1].set_title('T')

axarr[0,2].imshow(tqu_len.qmap)
axarr[0,2].get_xaxis().set_visible(False)
axarr[0,2].get_yaxis().set_visible(False)
axarr[0,2].set_title('Q')

axarr[0,3].imshow(tqu_len.umap)
axarr[0,3].get_xaxis().set_visible(False)
axarr[0,3].get_yaxis().set_visible(False)
axarr[0,3].set_title('U')

axarr[1,0].imshow(phihat_tt.get_rffts()[0].get_rmap().map)
axarr[1,0].get_xaxis().set_visible(False)
axarr[1,0].get_yaxis().set_visible(False)
axarr[1,0].set_title('Phi TT')

axarr[1,1].imshow(phihat_ee.get_rffts()[0].get_rmap().map)
axarr[1,1].get_xaxis().set_visible(False)
axarr[1,1].get_yaxis().set_visible(False)
axarr[1,1].set_title('Phi TE')

axarr[1,2].imshow(phihat_eb.get_rffts()[0].get_rmap().map)
axarr[1,2].get_xaxis().set_visible(False)
axarr[1,2].get_yaxis().set_visible(False)
axarr[1,2].set_title('Phi EB')

axarr[1,3].imshow(np.real(np.fft.ifft2(phihat_tt.fft*np.sqrt((phihat_tt.nx*phihat_tt.ny)/(phihat_tt.dx*phihat_tt.dy)))))#kappa_true.map)
axarr[1,3].get_xaxis().set_visible(False)
axarr[1,3].get_yaxis().set_visible(False)
axarr[1,3].set_title('Kappa True')

pdb.set_trace()

