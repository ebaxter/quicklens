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
from quicklens import sims
from quicklens import maps
from astropy.cosmology import Planck13 as cosmo
from astropy import units as u
from astropy import constants as const

import quicklens as ql

def get_NFW_kappa(pix, M_200, c_200, z_L, z_S):
    #Return a kappa map (as an rmap object) corresponding
    #to an NFW cluster observed across an angular window
    #specified by the pix object, pix
    #From http://arxiv.org/pdf/astro-ph/9908213v1.pdf
    rho_c_z = cosmo.critical_density(z_L)
    delta_c = (200./3.)*(c_200**3.)/(np.log(1.+c_200)-c_200/(1.+c_200))
    r_200 = (((M_200/(200.*4.*np.pi/3.))/rho_c_z)**(1./3.)).to('Mpc')
    r_s = r_200/c_200

    #Angular diameter distances
    D_L = cosmo.comoving_distance(z_L)/(1.+z_L)
    D_S = cosmo.comoving_distance(z_S)/(1.+z_S)
    D_LS = (cosmo.comoving_distance(z_S) - cosmo.comoving_distance(z_L))/(1.+z_S)
    Sigma_c = (((const.c.cgs**2.)/(4.*np.pi*const.G.cgs))*(D_S/(D_L*D_LS))).to('M_sun/Mpc2')

    #Angle between pixel center and center of map
    theta_x_list = np.array(range(pix.nx))*pix.dx-pix.dx*0.5*(pix.nx - 1.)
    theta_x = np.tile(theta_x_list,(pix.nx,1))
    theta_y = np.transpose(theta_x)
    theta = np.sqrt(theta_x**2. + theta_y**2.)
    R = D_L*theta
    x = R/r_s

    g_theta = np.zeros(x.shape)
    gt_one = np.where(x > 1.0)
    lt_one = np.where(x < 1.0)
    eq_one = np.where(x == 1.0)
    g_theta[gt_one] = (1./(x[gt_one]**2. - 1))*(1. - (2./np.sqrt(x[gt_one]**2. - 1.))*np.arctan(np.sqrt((x[gt_one]-1.)/(x[gt_one]+1.))).value)
    g_theta[lt_one] = (1./(x[lt_one]**2. - 1))*(1. - (2./np.sqrt(1. - x[lt_one]**2.))*np.arctanh(np.sqrt((1. - x[lt_one])/(x[lt_one]+1.))).value)
    g_theta[eq_one] = 1./3.

    Sigma = ((2.*r_s*delta_c*rho_c_z)*g_theta).to('M_sun/Mpc2')

    kappa = Sigma/Sigma_c
    kappa_rmap = ql.maps.rmap(pix.nx, pix.dx, map = kappa)
    return kappa_rmap

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

mask       = np.ones( (nx, nx) )
# mask to apply when inverse-variance filtering.
# currently, no masking.
# alternatively, cosine masking:
# x, y = np.meshgrid( np.arange(0,nx), np.arange(0,nx) )
# mask = np.sin( np.pi/nx*x )*np.sin( np.pi/nx*y )

mc_sims_mf = None
# indices of simulations to use for estimating a mean-field.
# currently, no mean-field subtraction.
# alternatively: np.arange(nsims, 2*nsims)

npad       = 1

#Cluster properties
M_200 = (1.e17)*u.M_sun
c_200 = 3.0
z_L = 1.0
z_S = 1089. #IS THERE SOME BETTER WAY TO GET THIS?  FROM ASTROPY?
cluster_kappa = get_NFW_kappa(pix, M_200, c_200, z_L, z_S)
#check this! why 10000????? should this be lmax?
#divide by 0?????? because of arange
#Is this the right way to do this?
#77777 something weird going on here, phi is very very large.
#related to value at l = 0?
temp = np.arange(0., 10000.)
temp[0]= 1.0e-1
cluster_phi_fft = cluster_kappa.get_rfft()*( 1./(0.5 * temp**2 ))
#cluster_phi_fft.fft = np.nan_to_num(cluster_phi_fft.fft)
#this is slightly weird 7777777
#why isn't this exactly equal to input kappa?
#Mean issue?
#Apodize?
fl = 1.0
test_kappa = ( cluster_phi_fft  * ( 0.5 * np.arange(0,10000)**2 ) * fl ).get_rmap()
test_phi = cluster_phi_fft.get_rmap()

#Get lensed and unlensed C_l's
cl_unl     = ql.spec.get_camb_scalcl(lmax=lmax)
cl_len     = ql.spec.get_camb_lensedcl(lmax=lmax)

#IS LENSING METHOD ACCURATE EVEN FOR HIGH KAPPA?  AM I CAPTURING FULL EFFECTS????77777

# make libraries for simulated skies.
#can we do this with not-flat sky?
sky_lib    = ql.sims.cmb.library_flat_lensed_fixphi(pix, cl_unl, cluster_phi_fft, "temp1/sky")
obs_lib    = ql.sims.obs.library_white_noise(pix, bl, sky_lib, nlev_t=nlev_t, nlev_p=nlev_p, lib_dir="temp1/obs")

# make libraries for inverse-variance filtered skies.
cl         = ql.spec.clmat_teb( ql.util.dictobj( {'lmax' : lmax, 'cltt' : cl_len.cltt, 'clee' : cl_len.clee, 'clbb' : cl_len.clbb} ) )
transf     = ql.spec.cl2tebfft(ql.util.dictobj( {'lmax' : lmax, 'cltt' : bl, 'clee' : bl, 'clbb' : bl} ), pix)
ivf_lib    = ql.sims.ivf.library_l_mask( ql.sims.ivf.library_diag(obs_lib, cl=cl, transf=transf, nlev_t=nlev_t, nlev_p=nlev_p, mask=mask), lmin=lmin, lmax=lmax )

#7777NEED TO DO MEAN FIELD SUBTRACTION TOO!!!!!

#Use library function to get quadratic estimators
qest_lib = ql.sims.qest.library(cl_unl, cl_len, ivf_lib, lib_dir="temp1/qest", npad=npad)

#For plotting
tqu_len = sky_lib.get_sim_tqu(0)
phi_true = sky_lib.get_sim_phi(0)
#this doesn't perfectly match input kappa, see above
kappa_true = sky_lib.get_sim_kappa(0)

estimator = 'ptt'
qft = qest_lib.get_sim_qft(estimator, 0)
qr = qest_lib.get_qr(estimator)
phihat_tt = qft / qr
phihat_tt.fft = np.nan_to_num(phihat_tt.fft)
#are we losing something by going from cfft to rfft?
fl = 1.0
kappahat_tt = (( phihat_tt  * ( 0.5 * np.arange(0,10000)**2 ) * fl ).get_rffts())[0].get_rmap()
                    
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
fig, axarr = pl.subplots(2,5)
fig.suptitle('CMB Lensing')

axarr[0,0].imshow(kappa_true.map)
axarr[0,0].get_xaxis().set_visible(False)
axarr[0,0].get_yaxis().set_visible(False)
axarr[0,0].set_title('True Kappa')

axarr[0,1].imshow(phi_true.map)
axarr[0,1].get_xaxis().set_visible(False)
axarr[0,1].get_yaxis().set_visible(False)
axarr[0,1].set_title('True Phi')

axarr[0,2].imshow(tqu_len.tmap)
axarr[0,2].get_xaxis().set_visible(False)
axarr[0,2].get_yaxis().set_visible(False)
axarr[0,2].set_title('T')

axarr[0,3].imshow(tqu_len.qmap)
axarr[0,3].get_xaxis().set_visible(False)
axarr[0,3].get_yaxis().set_visible(False)
axarr[0,3].set_title('Q')

axarr[0,4].imshow(tqu_len.umap)
axarr[0,4].get_xaxis().set_visible(False)
axarr[0,4].get_yaxis().set_visible(False)
axarr[0,4].set_title('U')

axarr[1,0].imshow(kappahat_tt.map)
axarr[1,0].get_xaxis().set_visible(False)
axarr[1,0].get_yaxis().set_visible(False)
axarr[1,0].set_title('kappa TT')

axarr[1,1].imshow(phihat_tt.get_rffts()[0].get_rmap().map)
axarr[1,1].get_xaxis().set_visible(False)
axarr[1,1].get_yaxis().set_visible(False)
axarr[1,1].set_title('Phi TT')

axarr[1,2].imshow(phihat_ee.get_rffts()[0].get_rmap().map)
axarr[1,2].get_xaxis().set_visible(False)
axarr[1,2].get_yaxis().set_visible(False)
axarr[1,2].set_title('Phi EE')

axarr[1,3].imshow(phihat_eb.get_rffts()[0].get_rmap().map)
axarr[1,3].get_xaxis().set_visible(False)
axarr[1,3].get_yaxis().set_visible(False)
axarr[1,3].set_title('Phi EB')


pdb.set_trace()

