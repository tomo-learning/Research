import numpy as np
from scipy import fftpack as ffp
import matplotlib.pyplot as plt

def rect(x):
    return np.where(np.abs(x) <= 0.5, 1 , 0)

N=4096 # サンプリング数
wavelen=0.6328*10**(-3) # 波長(mm)
dx=1e-3 # サンプリング間隔
f=2 # レンズの焦点距離
distance = np.arange(0, 5 + dx, dx) #レンズからの距離

x=np.linspace(-N/2,N/2-1,N)
gx=rect((x)/10000000) # 入射波面の振幅分布
# for i in range(len(gx)):
#     if i==1700:
#        gx[i]=1.0
#     else:
#        gx[i]=0.0

fx = np.zeros_like(gx, dtype=np.complex128)

# レンズの変換を適用
for i in range(N):
    fx[i]=gx[i]*np.exp(-1j*np.pi*(x[i]*dx)**2/(wavelen*f))
    if x[i]*dx==3.1:
        print(x[i])

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
    if z==2:
        image=amplitude_map[i,:]

fig,ax=plt.subplots(figsize=(5,4))
ax.plot(x,image)
ax.set_xlabel("x")
plt.show()

x1=x*dx # 実空間の座標

for i in range(len(distance)):
    if np.max(amplitude_map[i, :]) != 0:
        amplitude_map[i, :] /= np.max(amplitude_map[i, :])  # 各距離で正規化

# 振幅強度のマップを表示
fig1, ax1 = plt.subplots(figsize=(8, 6))
extent = [distance[0], distance[-1], x[0], x[-1]]
im = ax1.imshow(amplitude_map.T, extent=extent, origin='lower', aspect='auto', cmap='gray')  # カラーマップを 'gray' に変更
ax1.set_xlabel("$z$")
ax1.set_ylabel("$x$") 
fig1.colorbar(im, ax=ax1, label="Amplitude Intensity")
fig1.savefig("ASM1d_amplitude_map.png")
plt.show()

