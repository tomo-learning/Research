import numpy as np
from scipy import fftpack as ffp
import matplotlib.pyplot as plt
import copy

def rect(x):
    return np.where(np.abs(x) <= 0.5, 1 , 0)

def pad(u, m):
    up = np.zeros(len(u)+2*m, dtype=complex)
    up[m:m+len(u)] = u
    return up

def crop_center(u, N):
    s = (len(u)-N)//2
    return u[s:s+N]

N=4096*4 # サンプリング数
wavelen=532*10**(-6) # 波長(mm)
Npad=2048*8 # パディングサイズ
dx=0.25e-3 # サンプリング間隔
f=1 # レンズの焦点距離
distance = np.arange(0, 8 + dx, dx) #レンズからの距離

x=np.linspace(-N/2,N/2-1,N)
xpos=N//2+8000 #物体点の分布位置

#物体面の振幅分布
u0=[0.0 for _ in range(len(x))]
u0[xpos]=1

a=2 #1枚目のレンズと物体の距離
b=4 #レンズ間の距離
c=2 #2枚目のレンズと像の距離

x_mm = (np.arange(N) - N//2) * dx
D1 = 1.5  # 1枚目のレンズの直径(mm)
D2 = 1.5  # 2枚目のレンズの直径(mm)

# # ASMによるレンズ前面の波面の計算
# nux=ffp.fftfreq(N,dx)
# nu_sq=1/wavelen**2-nux**2
# mask=nu_sq>0
# phase_func=np.zeros(len(nux),dtype=np.complex128)
# phase_func[mask]=np.exp(1j*2*np.pi*np.sqrt(nu_sq[mask])*a)
# ux=ffp.ifft(ffp.fft(u0)*phase_func)

nux=ffp.fftfreq(N+Npad*2,dx)
nu_sq=1/wavelen**2-nux**2
mask=nu_sq>0

amplitude_map = np.zeros((len(distance), N)) #伝播の様子の可視化用の配列
pre_diffraction=np.zeros(N,dtype=np.complex128)
pre_diffraction=copy.copy(u0)
#物体面と一枚目のレンズ面の間の波面を各距離で計算
for i in range(len(distance)):
    if distance[i]>a:
        break
    phase_func=np.zeros(len(nux),dtype=np.complex128)
    phase_func[mask]=np.exp(1j*2*np.pi*np.sqrt(nu_sq[mask])*dx)
    padprediff=pad(pre_diffraction,Npad)
    diffraction=crop_center(ffp.ifft(ffp.fft(padprediff)*phase_func),N)
    pre_diffraction=diffraction
    amplitude_map[i, :] = np.abs(diffraction)**2
ux=copy.copy(pre_diffraction)

Px1=rect(x_mm / D1) # 一枚目レンズの瞳関数
fx = np.zeros_like(Px1, dtype=np.complex128)

# レンズの変換を適用
for i in range(N):
    fx[i]=ux[i]*Px1[i]*np.exp(-1j*np.pi*(x[i]*dx)**2/(wavelen*f))

#一枚目のレンズと二枚目のレンズの間の波面を各距離で計算
pre_diffraction=fx
for i in range(len(distance)):
    if not a<distance[i]<a+b:
        continue
    phase_func=np.zeros(len(nux),dtype=np.complex128)
    phase_func[mask]=np.exp(1j*2*np.pi*np.sqrt(nu_sq[mask])*dx)
    padprediff=pad(pre_diffraction,Npad)
    diffraction=crop_center(ffp.ifft(ffp.fft(padprediff)*phase_func),N)
    pre_diffraction=diffraction
    amplitude_map[i, :] = np.abs(diffraction)**2
gx=pre_diffraction

#二枚目のレンズ前面の波面を計算
# nux=ffp.fftfreq(N,dx)
# nu_sq=1/wavelen**2-nux**2
# mask=nu_sq>0
# phase_func=np.zeros(len(nux),dtype=np.complex128)
# phase_func[mask]=np.exp(1j*2*np.pi*np.sqrt(nu_sq[mask])*b)
# gx=ffp.ifft(ffp.fft(fx)*phase_func)


Px2=rect(x_mm / D2) # 二枚目のレンズの瞳関数
hx = np.zeros_like(Px2, dtype=np.complex128)

# レンズの変換を適用
for i in range(N):
    hx[i]=gx[i]*Px2[i]*np.exp(-1j*np.pi*(x[i]*dx)**2/(wavelen*f))

pre_diffraction=hx
#二枚目のレンズと結像面の間の波面を各距離で計算
for i in range(len(distance)):
    if distance[i]<=a+b:
        continue
    phase_func=np.zeros(len(nux),dtype=np.complex128)
    phase_func[mask]=np.exp(1j*2*np.pi*np.sqrt(nu_sq[mask])*dx)
    padprediff=pad(pre_diffraction,Npad)
    diffraction=crop_center(ffp.ifft(ffp.fft(padprediff)*phase_func),N)
    pre_diffraction=diffraction
    amplitude_map[i, :] = np.abs(diffraction)**2
image=amplitude_map[-1,:]
#結像面での波面を計算
# nux=ffp.fftfreq(N,dx)
# nu_sq=1/wavelen**2-nux**2
# mask=nu_sq>0
# phase_func=np.zeros(len(nux),dtype=np.complex128)
# phase_func[mask]=np.exp(1j*2*np.pi*np.sqrt(nu_sq[mask])*c)
# diffraction=ffp.ifft(ffp.fft(hx)*phase_func)
# image=np.abs(diffraction)**2


x1=x*dx # 実空間の座標

#結像面の強度分布をプロット
fig,ax=plt.subplots(figsize=(5,4))
ax.plot(x,image)
ax.set_xlabel("x")
plt.show()

for i in range(len(distance)):
    if np.max(amplitude_map[i, :]) != 0:
        amplitude_map[i, :] /= np.max(amplitude_map[i, :])  # 各距離で正規化

        
#各距離における強度のマップを表示
fig1, ax1 = plt.subplots(figsize=(16, 8))
extent = [distance[0], distance[-1], x1[0], x1[-1]]
im = ax1.imshow(amplitude_map.T, extent=extent, origin='lower', aspect='auto', cmap='gray')  # カラーマップを 'gray' に変更
ax1.set_xlabel("$z$")
ax1.set_ylabel("$x$") 
fig1.colorbar(im, ax=ax1, label="Amplitude Intensity")
fig1.savefig("Erectimage_amplitude_map.png")
plt.show()

