import numpy as np
import matplotlib.pyplot as plt

def rect2d(x,y):
    return np.where((np.abs(x) <= 0.5) & (np.abs(y) <= 0.5), 1 , 0)


N = 512
N2 = N/2
wavelen=0.6328*10**(-3)
dx,dy=1.0e-3,1.0e-3

x=np.arange(N)
y=np.arange(N)
X,Y=np.meshgrid(x,y)
Z=rect2d((X-N2-64)/64,(Y-N2)/256) + rect2d((X-N2+64)/64,(Y-N2)/256)

fig,ax=plt.subplots(2,4,figsize=(14,6))
ax[0,0].plot(Z[256,:],"-k")
ax[0,0].set_title("z=0.0 mm")
ax[1,0].imshow(Z,cmap="gray",origin="lower")

nux=np.fft.fftfreq(N,dx)
nuy=np.fft.fftfreq(N,dy)
NUX,NUY=np.meshgrid(nux,nuy)
nu_sq=1/wavelen**2-(NUX**2+NUY**2)
mask=nu_sq>0
weight=np.zeros((N,N),dtype=np.complex128)

distance=[0.1,0.25,1]
for i in range(3):
    z=distance[i]
    weight[mask]=np.exp(1j*2*np.pi*np.sqrt(nu_sq[mask])*z)

    diffraction=np.fft.ifft2(weight*np.fft.fft2(Z))
    ax[0,i+1].plot(np.abs(diffraction[256,:])**2,"-k")
    ax[0,i+1].set_title("z={} mm".format(z))
    ax[1,i+1].imshow(np.abs(diffraction)**2,cmap="gray",origin="lower")

plt.savefig("program6-2.png")