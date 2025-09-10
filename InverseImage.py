import numpy as np
from scipy import fftpack as ffp
import matplotlib.pyplot as plt
import cmath
import math

def rect(x):
    return np.where(np.abs(x) <= 0.5, 1 , 0)

N=4096 # サンプリング数
wavelen=0.6328*10**(-3) # 波長(mm)
dx=0.5e-3 # サンプリング間隔
f=1.0 # レンズの焦点距離
distance = np.arange(0, 10 + dx, dx) #レンズからの距離
print(wavelen)

x=np.linspace(-N/2,N/2-1,N)
xpos=2048
#物体面の振幅分布
u0=[0.0 for _ in range(len(x))]
for i in range(len(u0)):
    if i==xpos:
       u0[i]=1.0
a=2 #レンズと物体の距離

# ASMによるレンズ前面の波面の計算
nux=ffp.fftfreq(N,dx)
nu_sq=1/wavelen**2-nux**2
mask=nu_sq>0
phase_func=np.zeros(len(nux),dtype=np.complex128)
phase_func[mask]=np.exp(1j*2*np.pi*np.sqrt(nu_sq[mask])*a)
ux=ffp.ifft(ffp.fft(u0)*phase_func)
image=np.abs(ux)**2

Px=rect((x)/1024) # 瞳関数
fx = np.zeros_like(Px, dtype=np.complex128)

# レンズの変換を適用
for i in range(N):
    fx[i]=ux[i]*Px[i]*np.exp(-1j*np.pi*(x[i]*dx)**2/(wavelen*f))
#image=np.abs(fx)**2
print(ux[2188])
print(fx[2188])
print(math.degrees(math.atan2((xpos-N/2)*dx,a)))

amplitude_map = np.zeros((len(distance), N))

# 各距離で角スペクトル法による伝搬計算
for i in range(len(distance)):
    z=distance[i]
    nux=ffp.fftfreq(N,dx)
    nu_sq=1/wavelen**2-nux**2
    mask=nu_sq>0
    phase_func=np.zeros(len(nux),dtype=np.complex128)
    phase_func[mask]=np.exp(1j*2*np.pi*np.sqrt(nu_sq[mask])*z)
    diffraction=ffp.ifft(ffp.fft(fx)*phase_func)
    amplitude_map[i, :] = np.abs(diffraction)**2
    # if z==0.001:
    #     image=amplitude_map[i,:]


x1=x*dx # 実空間の座標

for i in range(len(distance)):
    if np.max(amplitude_map[i, :]) != 0:
        amplitude_map[i, :] /= np.max(amplitude_map[i, :])  # 各距離で正規化

fig,ax=plt.subplots(figsize=(5,4))
ax.plot(x,image)
ax.set_xlabel("x")
plt.show()


# 振幅強度のマップを表示
fig1, ax1 = plt.subplots(figsize=(8, 6))
extent = [distance[0], distance[-1], x[0], x[-1]]
im = ax1.imshow(amplitude_map.T, extent=extent, origin='lower', aspect='auto', cmap='gray')  # カラーマップを 'gray' に変更
ax1.set_xlabel("$z$")
ax1.set_ylabel("$x$") 
fig1.colorbar(im, ax=ax1, label="Amplitude Intensity")
fig1.savefig("Erectimage_amplitude_map.png")
plt.show()

