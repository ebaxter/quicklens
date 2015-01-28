#/usr/bin/env python
# --
# quicklens/examples/lens/get_phi_map.py
# --
# generates a set of lensed maps in the flat-sky limit, then runs
# quadratic lensing estimators on them to estimate phi.
# Plots map of reconstructed phi.  EB.

import numpy as np
import matplotlib.pyplot as pl
import pdb
from quicklens import sims
from quicklens import maps
from astropy.cosmology import Planck13 as cosmo
from astropy import units as u
from astropy import constants as const
from scipy.interpolate import RectBivariateSpline

import quicklens as ql

def get_NFW_kappa(theta, mass, concentration, z_L, z_S, mass_def, rho_def):
    """
    Purpose: return kappa given input theta for a NFW cluster
    observed across a specified angular window.  Formulae from
    http://arxiv.org/pdf/astro-ph/9908213v1.pdf.

    Input:
    theta = angle from center of cluster in radians (can be array)
    mass = mass of cluster in given mass_def and rho_def convention
    concentration = concentration of cluser in given mass_def and rho_def convention
    z_L = redshift of lens
    z_S = redshift of source
    mass_def = factor by which average density in cluster exceeds density of universe (where density of universe definition depends on rho_def).  Common values are 200 and 500
    rho_def = either 'crit' or 'mean'

    Output:
    kappa = kappa values evaluated at theta

    THIS DIVERGES AT THETA = 0
    """
    #7777 CHECK THIS
    if (rho_def == 'crit'):
        rho_c_z = cosmo.critical_density(z_L)
    elif (rho_def == 'mean'):
        rho_c_z = cosmo.Om(z_L)*cosmo.critical_density(z_L)
    else:
        print "rho definition not specified correctly in cluster profile"
        assert(0)
    #NFW profile properties
    delta_c = (mass_def/3.)*(concentration**3.)/(np.log(1.+concentration)-concentration/(1.+concentration))
    r_v = (((mass/(mass_def*4.*np.pi/3.))/rho_c_z)**(1./3.)).to('Mpc')
    r_s = r_v/concentration
    #Angular diameter distances
    D_L = cosmo.comoving_distance(z_L)/(1.+z_L)
    D_S = cosmo.comoving_distance(z_S)/(1.+z_S)
    D_LS = (cosmo.comoving_distance(z_S)-cosmo.comoving_distance(z_L))/(1.+z_S)
    #Normalization of kappa
    Sigma_c = (((const.c.cgs**2.)/(4.*np.pi*const.G.cgs))*(D_S/(D_L*D_LS))).to('M_sun/Mpc2')
    #Useful variables
    R = D_L*theta
    x = R/r_s
    #Functional form of profile
    g_theta = np.zeros(x.shape)
    gt_one = np.where(x > 1.0)
    lt_one = np.where(x < 1.0)
    eq_one = np.where(x == 1.0)
    g_theta[gt_one] = (1./(x[gt_one]**2. - 1))*(1. - (2./np.sqrt(x[gt_one]**2. - 1.))*np.arctan(np.sqrt((x[gt_one]-1.)/(x[gt_one]+1.))).value)
    g_theta[lt_one] = (1./(x[lt_one]**2. - 1))*(1. - (2./np.sqrt(1. - x[lt_one]**2.))*np.arctanh(np.sqrt((1. - x[lt_one])/(x[lt_one]+1.))).value)
    g_theta[eq_one] = 1./3.
    #Projected mass
    Sigma = ((2.*r_s*delta_c*rho_c_z)*g_theta).to('M_sun/Mpc2')
    #Find kappa
    kappa = Sigma/Sigma_c
    return kappa

def get_NFW_kappa_pix_grid(pix, mass, concentration, z_L, z_S, mass_def, rho_def):
    """
    Purpose: return kappa map as maps.rmap object given input maps.pix object

    Input:
    pix = maps.pix object specifying angular window
    mass = mass of cluster in given mass_def and rho_def convention
    concentration = concentration of cluser in given mass_def and rho_def convention
    z_L = redshift of lens
    z_S = redshift of source
    mass_def = factor by which average density in cluster exceeds density of universe (where density of universe definition depends on rho_def).  Common values are 200 and 500
    rho_def = either 'crit' or 'mean'

    Output:
    kappa = maps.rmap object containing kappa map
    """
    #Angle between pixel center and center of map
    theta_x_list = np.array(range(pix.nx))*pix.dx-pix.dx*0.5*(pix.nx - 1.)
    theta_x = np.tile(theta_x_list,(pix.nx,1))
    theta_y = np.transpose(theta_x)
    theta = np.sqrt(theta_x**2. + theta_y**2.)
    #Get kappa values
    kappa = get_NFW_kappa(theta, mass, concentration, z_L, z_S, mass_def, rho_def)
    #Convert to maps.rmap object
    kappa_rmap = ql.maps.rmap(pix.nx, pix.dx, map = kappa)
    return kappa_rmap

# simulation parameters.
nsims      = 1
lmin       = 10
lmax       = 9999
nx         = 512 # number of pixels.
dx         = 0.2/60./180.*np.pi # pixel width in radians.

nlev_t     = 0.0  # temperature noise level, in uK.arcmin.
nlev_p     = np.sqrt(2.0)*nlev_t  # polarization noise level, in uK.arcmin.
bl         = ql.spec.bl(0., lmax) # beam transfer function.

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
M_200 = (1.e16)*u.M_sun
c_200 = 3.0
mass_def = 200.0
rho_def = 'crit'
z_L = 0.7
z_S = 1090.43
#Compute kappa from NFW cluster
cluster_kappa = get_NFW_kappa_pix_grid(pix, M_200, c_200, z_L, z_S, mass_def, rho_def)
#Convert kappa into Fourier transform of phi
#In Fourier space, phi = kappa / (0.5*ell**2)
#This may not be 100% correct!
temp = np.arange(0., 20000.)
temp[0]= 1.0
to_multiply = ( 1./(0.5 * temp**2 ))
to_multiply[0] = 0.0
cluster_phi_fft = cluster_kappa.get_rfft()*to_multiply
#Test to make sure we recover input kappa
#This is slightly confusing since we lose information about mean
#So we only expect test_kappa to match cluster_kappa up to an additive constant
fl = 1.0
test_kappa = ( cluster_phi_fft  * ( 0.5 * np.arange(0,10000)**2 ) * fl ).get_rmap()

#Get lensed and unlensed C_l's
cl_unl     = ql.spec.get_camb_scalcl(lmax=lmax)
cl_len     = ql.spec.get_camb_lensedcl(lmax=lmax)

# make libraries for simulated skies.
#can we do this with not-flat sky?
sky_lib    = ql.sims.cmb.library_flat_lensed_fixphi(pix, cl_unl, cluster_phi_fft, "temp/sky")
obs_lib    = ql.sims.obs.library_white_noise(pix, bl, sky_lib, nlev_t=nlev_t, nlev_p=nlev_p, lib_dir="temp/obs")

# make libraries for inverse-variance filtered skies.
cl         = ql.spec.clmat_teb( ql.util.dictobj( {'lmax' : lmax, 'cltt' : cl_len.cltt, 'clee' : cl_len.clee, 'clbb' : cl_len.clbb} ) )
transf     = ql.spec.cl2tebfft(ql.util.dictobj( {'lmax' : lmax, 'cltt' : bl, 'clee' : bl, 'clbb' : bl} ), pix)
ivf_lib    = ql.sims.ivf.library_l_mask( ql.sims.ivf.library_diag(obs_lib, cl=cl, transf=transf, nlev_t=nlev_t, nlev_p=nlev_p, mask=mask), lmin=lmin, lmax=lmax )

#Use library function to get quadratic estimators
qest_lib = ql.sims.qest.library(cl_unl, cl_len, ivf_lib, lib_dir="temp/qest", npad=npad)

#For plotting
tqu_len = sky_lib.get_sim_tqu(0)
phi_true = sky_lib.get_sim_phi(0)
#this doesn't perfectly match input kappa, see above
kappa_true = sky_lib.get_sim_kappa(0)

#Compute quadratic estimators of phi field using different observables
#For TT, also compute kappa
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

######################################################################
#Everything below here is for plotting
fig, axarr = pl.subplots(2,4)

#For axes
theta_x_list = np.array(range(pix.nx))*pix.dx-pix.dx*0.5*(pix.nx - 1.)
theta_x_list_arcmin = theta_x_list*(180./np.pi)*60.

kappa_min_level = np.min(kappahat_tt.map)
kappa_max_level = np.max(kappahat_tt.map)
num_kappa_levels = 20
kappa_levels = np.linspace(kappa_min_level, kappa_max_level, num = num_kappa_levels, endpoint = True)

phi_min_level = np.min(phihat_tt.get_rffts()[0].get_rmap().map)
phi_max_level = np.max(phihat_tt.get_rffts()[0].get_rmap().map)
num_phi_levels = 20
phi_levels = np.linspace(phi_min_level, phi_max_level, num = num_phi_levels, endpoint = True)

axarr[0,0].contourf(theta_x_list_arcmin, theta_x_list_arcmin, kappa_true.map, kappa_levels)
axarr[0,0].get_xaxis().set_visible(False)
axarr[0,0].get_yaxis().set_visible(False)
axarr[0,0].set_title('True Kappa')

axarr[0,1].contourf(theta_x_list_arcmin, theta_x_list_arcmin, phi_true.map, phi_levels)
axarr[0,1].get_xaxis().set_visible(False)
axarr[0,1].get_yaxis().set_visible(False)
axarr[0,1].set_title('True Phi')

axarr[0,2].contourf(theta_x_list_arcmin, theta_x_list_arcmin,tqu_len.tmap, 20)
axarr[0,2].get_xaxis().set_visible(False)
axarr[0,2].get_yaxis().set_visible(False)
axarr[0,2].set_title('T')

axarr[0,3].contourf(theta_x_list_arcmin, theta_x_list_arcmin,tqu_len.qmap, 20)
axarr[0,3].get_xaxis().set_visible(False)
axarr[0,3].get_yaxis().set_visible(False)
axarr[0,3].set_title('Q')

axarr[1,0].contourf(theta_x_list_arcmin, theta_x_list_arcmin, kappahat_tt.map, kappa_levels)
axarr[1,0].get_xaxis().set_visible(False)
axarr[1,0].get_yaxis().set_visible(False)
axarr[1,0].set_title('kappa TT')

axarr[1,1].contourf(theta_x_list_arcmin, theta_x_list_arcmin,phihat_tt.get_rffts()[0].get_rmap().map, phi_levels)
axarr[1,1].get_xaxis().set_visible(False)
axarr[1,1].get_yaxis().set_visible(False)
axarr[1,1].set_title('Phi TT')

axarr[1,2].contourf(theta_x_list_arcmin, theta_x_list_arcmin,phihat_ee.get_rffts()[0].get_rmap().map, phi_levels)
axarr[1,2].get_xaxis().set_visible(False)
axarr[1,2].get_yaxis().set_visible(False)
axarr[1,2].set_title('Phi EE')

axarr[1,3].contourf(theta_x_list_arcmin, theta_x_list_arcmin,phihat_eb.get_rffts()[0].get_rmap().map, phi_levels)
axarr[1,3].get_xaxis().set_visible(False)
axarr[1,3].get_yaxis().set_visible(False)
axarr[1,3].set_title('Phi EB')

fig.show()
