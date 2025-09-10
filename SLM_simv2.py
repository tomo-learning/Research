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

N=4096*32 # サンプリング数
dn=128
wavelen=532*10**(-6) # 波長(mm)
Npad=4096*16 # パディングサイズ
dx=1e-4 # サンプリング間隔
dz=0.5
f=1 # レンズの焦点距離

x=np.linspace(-N/2,N/2-1,N)
xpos=N//2+4000 #物体点の分布位置



a=2 #1枚目のレンズと物体の距離
b=4 #レンズ間の距離
c=2 #2枚目のレンズと像の距離

x_mm = (np.arange(N) - N//2) * dx
D1 = 1.5  # 1枚目のレンズの直径(mm)
D2 = 1.5  # 2枚目のレンズの直径(mm)


nux=ffp.fftfreq(N+Npad*2,dx)
nu_sq=1/wavelen**2-nux**2
mask=nu_sq>0
amplitude_map = np.zeros(N//dn)
xout=[]
for k in range(N//dn):
    xpos=k*dn
    xout.append(x[xpos]*dx)
    #物体面の振幅分布
    u0=[0.0 for _ in range(len(x))]
    u0[xpos]=1

    phase_func=np.zeros(len(nux),dtype=np.complex128)
    phase_func[mask]=np.exp(1j*2*np.pi*np.sqrt(nu_sq[mask])*a)
    upad=pad(u0,Npad)
    ux=crop_center(ffp.ifft(ffp.fft(upad)*phase_func),N)

    Px1=rect(x_mm / D1) # 一枚目レンズの瞳関数
    inputIntensity=0
    for i in range(len(ux)):
        if Px1[i]==1:
            inputIntensity+=np.abs(ux[i])**2
    if inputIntensity==0:
        continue
    fx = np.zeros_like(Px1, dtype=np.complex128)

    # レンズの変換を適用
    for i in range(N):
        fx[i]=ux[i]*Px1[i]*np.exp(-1j*np.pi*(x[i]*dx)**2/(wavelen*f))

    #一枚目のレンズと二枚目のレンズの間の波面を各距離で計算
    phase_func=np.zeros(len(nux),dtype=np.complex128)
    phase_func[mask]=np.exp(1j*2*np.pi*np.sqrt(nu_sq[mask])*b)
    fpad=pad(fx,Npad)
    fftfx=ffp.fft(fpad)
    gx=crop_center(ffp.ifft(fftfx*phase_func),N)


    Px2=rect(x_mm / D2) # 二枚目のレンズの瞳関数
    hx = np.zeros_like(Px2, dtype=np.complex128)

    # レンズの変換を適用
    for i in range(N):
        hx[i]=gx[i]*Px2[i]*np.exp(-1j*np.pi*(x[i]*dx)**2/(wavelen*f))


    #結像面での波面を計算
    phase_func=np.zeros(len(nux),dtype=np.complex128)
    phase_func[mask]=np.exp(1j*2*np.pi*np.sqrt(nu_sq[mask])*c)
    hpad=pad(hx,Npad)
    diffraction=crop_center(ffp.ifft(ffp.fft(hpad)*phase_func),N)
    amplitude_map[k] = sum(np.abs(diffraction)**2)/inputIntensity

    print(k/(N//dn)*100)


x1=x*dx # 実空間の座標

#像の表示
fig,ax=plt.subplots(figsize=(5,4))
ax.plot(xout,amplitude_map)
ax.set_xlabel("x")
plt.show()

