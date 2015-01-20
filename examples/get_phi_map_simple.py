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

import quicklens as ql

def init_q(k):
    qes        = {} # estimators
    if k == 'ptt':
        qes[k] = [ ( (ql.qest.lens.phi_TT(cl_len.cltt), 'TT'), 1.) ]
    elif k == 'pee':
        qes[k] = [ ( (ql.qest.lens.phi_EE(cl_len.clee), 'EE'), 1.) ]
    elif k == 'pte':
        qes[k] = [ ( (ql.qest.lens.phi_TE(cl_len.clte), 'TE'), 1.),
                   ( (ql.qest.lens.phi_ET(cl_len.clte), 'ET'), 1.) ]
    elif k == 'ptb':
        qes[k] = [ ( (ql.qest.lens.phi_TB(cl_len.clte), 'TB'), 1.),
                   ( (ql.qest.lens.phi_BT(cl_len.clte), 'BT'), 1.) ]
    elif k == 'peb':
        qes[k] = [ ( (ql.qest.lens.phi_EB(cl_len.clee), 'EB'), 1.),
                   ( (ql.qest.lens.phi_BE(cl_len.clee), 'BE'), 1.) ]
    elif k == 'pmv': # minimum-variance lensing estimator.
        qes[k] = [ ( 'ptt', 1. ),
                   ( 'pee', 1. ),
                   ( 'pte', 1. ),
                   ( 'ptb', 1. ),
                   ( 'peb', 1. ) ]
    return qes

def get_qr(ivfs1, ivfs2, estimator, npad, ks=None):
    """ return a maps.cfft object containing the response of a quadratic estimator 'ke' to another source of statistical anisotropy 'ks'.
    ke = estimator key.
    (optional) ks = source key (defaults to ke).
    """
    qrs        = {} # estimator responses
    ke = estimator
    if ks == None:
        ks = ke
        
    if (isinstance(ke, tuple) and isinstance(ks, tuple)): # a (qest.qest, scaling) pair for both estimator and source keys.
        qe, f12, f1, f2 = ke
        qs, f34 = ks
        
        if f12 == f34:
            return qe.fill_resp( qs, ql.maps.cfft(f1.nx, f1.dx, ny=f1.ny, dy=f1.dy), f1.fft, f2.fft, npad=npad)
        else:
            return ql.maps.cfft(f1.nx, f1.dx, ny=f1.ny, dy=f1.dy)

    elif isinstance(ke, tuple): # a (qest.qest, scaling) pair only for the estimator. need to expand the source key.
        qe, f12, f1, f2 = ke
        
        ret = ql.maps.cfft(f1.nx, f1.dx, ny=f1.ny, dy=f1.dy)
        
        #777777 check this
        for (tqs, tfs) in (init_q(ks))[ks]:
            ret += get_qr( ivfs1, ivfs2, ke, npad, ks=tqs) * tfs
        return ret
        
    else: # keys for both the estimator and source. need to expand.
        if (ke,ks) not in qrs.keys():
            tfl1, efl1, bfl1 = ivfs1.get_fl().get_cffts()
            if ivfs2 is not ivfs1:
                tfl2, efl2, bfl2 = ivfs2.get_fl().get_cffts()
            else:
                tfl2, efl2, bfl2 = tfl1, efl1, bfl1
                
            ret = ql.maps.cfft(tfl1.nx, tfl1.dx, ny=tfl1.ny, dy=tfl1.dy)
            #777777 check this
            for tqe, tfe in (init_q(ke))[ke]:
                if not isinstance(tqe, tuple):
                    ret += get_qr( ivfs1, ivfs2, tqe, npad, ks=ks ) * tfe
                else:
                    qe, f12 = tqe
                    
                    f1 = {'T' : tfl1, 'E' : efl1, 'B' : bfl1}[f12[0]]
                    f2 = {'T' : tfl2, 'E' : efl2, 'B' : bfl2}[f12[1]]

                    #77777 check this
                    for (tqs, tfs) in (init_q(ks))[ks]:
                        ret += get_qr(ivfs1, ivfs2, (qe,f12,f1,f2), npad, ks=tqs ) * tfe * tfs
            qrs[(ke,ks)] = ret
            return qrs[(ke,ks)]


def get_quad(ivfs1, ivfs2, estimator,npad):
    #index of sim???
    #777777 change this
    i = 0
    
    tft1, eft1, bft1 = ivfs1.get_sim_teb(i).get_cffts()
    if ivfs2 is not ivfs1:
        tft2, eft2, bft2 = ivfs2.get_sim_teb(i).get_cffts()
    else:
        tft2, eft2, bft2 = tft1, eft1, bft1
    k = estimator
    qes = init_q(k)
        
    ret = ql.maps.cfft(tft1.nx, tft1.dx, ny=tft1.ny, dy=tft1.dy)
    for tqe, tfe in qes[k]:
        (qe, f12) = tqe
        #got rid of shit here, what was it doing?
        f1 = {'T' : tft1, 'E' : eft1, 'B' : bft1}[f12[0]]
        f2 = {'T' : tft2, 'E' : eft2, 'B' : bft2}[f12[1]]
        ret += qe.eval(f1, f2, npad=npad) * tfe
##for tqe, tfe in self.get_qe(k):
##    if not isinstance(tqe, tuple):
##        ret += self.get_qft( tqe, tft1, eft1, bft1, tft2, eft2, bft2 ) * tfe
##    else:
##        (qe, f12) = tqe
##        
##        f1 = {'T' : tft1, 'E' : eft1, 'B' : bft1}[f12[0]]
##        f2 = {'T' : tft2, 'E' : eft2, 'B' : bft2}[f12[1]]
##        ret += qe.eval(f1, f2, npad=self.npad) * tfe
    my_qft = ret
    return my_qft

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

# plotting parameters.
t          = lambda l: (l+0.5)**4/(2.*np.pi) # scaling to apply to cl_phiphi when plotting.
lbins      = np.linspace(10, lmax, 20)       # multipole bins.

#Get lensed and unlensed C_l's
cl_unl     = ql.spec.get_camb_scalcl(lmax=lmax)
cl_len     = ql.spec.get_camb_lensedcl(lmax=lmax)
#Put 1-D power spectrum into 2D plane
#c_L_phiphi
clpp       = ql.spec.cl2cfft(cl_unl.clpp, ql.maps.cfft(nx,dx)).get_ml(lbins, t=t)

# make libraries for simulated skies.
#can we do this with not-flat sky?
sky_lib    = ql.sims.cmb.library_flat_lensed(pix, cl_unl, "temp6/sky")

teb_unl = sims.tebfft( pix, cl_unl )
#can change this to be whatever we want
phi_fft = sims.rfft( pix, cl_unl.clpp )

tqu_unl = teb_unl.get_tqu()
tqu_len = ql.lens.make_lensed_map_flat_sky( tqu_unl, phi_fft )

obs_lib    = ql.sims.obs.library_white_noise(pix, bl, sky_lib, nlev_t=nlev_t, nlev_p=nlev_p, lib_dir="temp6/obs")

#GET RID OF THIS LIBRARY SHIITE

# make libraries for inverse-variance filtered skies.
cl         = ql.spec.clmat_teb( ql.util.dictobj( {'lmax' : lmax, 'cltt' : cl_len.cltt, 'clee' : cl_len.clee, 'clbb' : cl_len.clbb} ) )
transf     = ql.spec.cl2tebfft(ql.util.dictobj( {'lmax' : lmax, 'cltt' : bl, 'clee' : bl, 'clbb' : bl} ), pix)
ivf_lib    = ql.sims.ivf.library_l_mask( ql.sims.ivf.library_diag(obs_lib, cl=cl, transf=transf, nlev_t=nlev_t, nlev_p=nlev_p, mask=mask), lmin=lmin, lmax=lmax )

#Get quadratic estimators from inside this prog
my_qft_tt = get_quad(ivf_lib, ivf_lib,'ptt',npad)
my_qr = get_qr(ivf_lib, ivf_lib, 'ptt', npad)
my_phihat_tt = my_qft_tt/my_qr
my_phihat_tt.fft = np.nan_to_num(my_phihat_tt.fft)

#Get quadratic estimators from inside this prog
my_qft_ee = get_quad(ivf_lib, ivf_lib,'pee',npad)
my_qr = get_qr(ivf_lib, ivf_lib, 'pee', npad)
my_phihat_ee = my_qft_ee/my_qr
my_phihat_ee.fft = np.nan_to_num(my_phihat_ee.fft)

#Get quadratic estimators from inside this prog
my_qft_eb = get_quad(ivf_lib, ivf_lib,'peb',npad)
my_qr = get_qr(ivf_lib, ivf_lib, 'peb', npad)
my_phihat_eb = my_qft_eb/my_qr
my_phihat_eb.fft = np.nan_to_num(my_phihat_eb.fft)

#For plotting
tqu_len = sky_lib.get_sim_tqu(0)
phi_true = sky_lib.get_sim_phi(0)
kappa_true = sky_lib.get_sim_kappa(0)

#7777NEED TO DO MEAN FIELD SUBTRACTION TOO!!!!!

#Use library function to get quadratic estimators
qest_lib = ql.sims.qest.library(cl_unl, cl_len, ivf_lib, lib_dir="temp6/qest", npad=npad)

estimator = 'ptt'
qft = qest_lib.get_sim_qft(estimator, 0)
qr = qest_lib.get_qr(estimator)
phihat_tt = qft / qr
phihat_tt.fft = np.nan_to_num(phihat_tt.fft)
                    
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
fig, axarr = pl.subplots(3,4)
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
axarr[1,1].set_title('Phi EE')

axarr[1,2].imshow(phihat_eb.get_rffts()[0].get_rmap().map)
axarr[1,2].get_xaxis().set_visible(False)
axarr[1,2].get_yaxis().set_visible(False)
axarr[1,2].set_title('Phi EB')

axarr[2,0].imshow(np.real(np.fft.ifft2(my_phihat_tt.fft*np.sqrt((my_phihat_tt.nx*my_phihat_tt.ny)/(my_phihat_tt.dx*my_phihat_tt.dy)))))
axarr[2,0].get_xaxis().set_visible(False)
axarr[2,0].get_yaxis().set_visible(False)
axarr[2,0].set_title('My Phi TT')

axarr[2,1].imshow(np.real(np.fft.ifft2(my_phihat_ee.fft*np.sqrt((my_phihat_ee.nx*my_phihat_ee.ny)/(my_phihat_ee.dx*my_phihat_ee.dy)))))
axarr[2,1].get_xaxis().set_visible(False)
axarr[2,1].get_yaxis().set_visible(False)
axarr[2,1].set_title('My Phi EE')

axarr[2,2].imshow(np.real(np.fft.ifft2(my_phihat_eb.fft*np.sqrt((my_phihat_eb.nx*my_phihat_eb.ny)/(my_phihat_eb.dx*my_phihat_eb.dy)))))
axarr[2,2].get_xaxis().set_visible(False)
axarr[2,2].get_yaxis().set_visible(False)
axarr[2,2].set_title('My Phi EB')




pdb.set_trace()

