import numpy as np
from scipy import fftpack as ffp
import matplotlib.pyplot as plt

def rect(x):
    return np.where(np.abs(x) <= 0.5, 1 , 0)

def pad(u, m):
    up = np.zeros(len(u)+2*m, dtype=complex)
    up[m:m+len(u)] = u
    return up

def crop_center(u, N):
    s = (len(u)-N)//2
    return u[s:s+N]

N=4096*8 # サンプリング数
Npad=2048*8 # パディングサイズ
wavelen=532*10**(-6) # 波長(mm)
dx=0.25e-3 # サンプリング間隔
f=1 # レンズの焦点距離

x=np.linspace(-N/2,N/2-1,N)
amplitude_map = np.zeros(len(x))
for k in range(N):
    #物体面の振幅分布
    xpos=k
    u0=[0.0 for _ in range(len(x))]
    for i in range(len(u0)):
        if i==xpos:
            u0[i]=1.0
    a=2 #1枚目のレンズと物体の距離
    b=4 #レンズ間の距離
    c=2 #2枚目のレンズと像の距離

    x_mm = (np.arange(N) - N//2) * dx
    D1 = 1.5  # 1枚目のレンズの直径(mm)
    D2 = 1.5  # 2枚目のレンズの直径(mm)

    # ASMによる一枚目のレンズ前面の波面の計算
    nux=ffp.fftfreq(N,dx)
    nu_sq=1/wavelen**2-nux**2
    mask=nu_sq>0
    phase_func=np.zeros(len(nux),dtype=np.complex128)
    phase_func[mask]=np.exp(1j*2*np.pi*np.sqrt(nu_sq[mask])*a)
    ux=ffp.ifft(ffp.fft(u0)*phase_func)



    Px1=rect(x_mm/D1) # 一枚目レンズの瞳関数
    fx = np.zeros_like(Px1, dtype=np.complex128)
    inputIntensity=0
    for i in range(len(ux)):
        if Px1[i]==1:
            inputIntensity+=np.abs(ux[i])**2

    # レンズの変換を適用
    for i in range(N):
        fx[i]=ux[i]*Px1[i]*np.exp(-1j*np.pi*(x[i]*dx)**2/(wavelen*f))


    # 倒立結像面での波面の計算
    phase_func=np.zeros(len(nux),dtype=np.complex128)
    phase_func[mask]=np.exp(1j*2*np.pi*np.sqrt(nu_sq[mask])*a)
    diffraction=ffp.ifft(ffp.fft(fx)*phase_func)
    image=np.abs(diffraction)**2

    # ASMによる二枚目のレンズ前面の波面の計算
    phase_func=np.zeros(len(nux),dtype=np.complex128)
    phase_func[mask]=np.exp(1j*2*np.pi*np.sqrt(nu_sq[mask])*b)


    # ASMによる二枚目のレンズ前面の波面の計算
    phase_func=np.zeros(len(nux),dtype=np.complex128)
    phase_func[mask]=np.exp(1j*2*np.pi*np.sqrt(nu_sq[mask])*b)
    gx=ffp.ifft(ffp.fft(fx)*phase_func)

    Px2=rect(x_mm / D2) # 二枚目レンズの瞳関数
    hx = np.zeros_like(Px2, dtype=np.complex128)

    # レンズの変換を適用
    for i in range(N):
        hx[i]=gx[i]*Px2[i]*np.exp(-1j*np.pi*(x[i]*dx)**2/(wavelen*f))


    # 像面での波面の計算
    phase_func=np.zeros(len(nux),dtype=np.complex128)
    phase_func[mask]=np.exp(1j*2*np.pi*np.sqrt(nu_sq[mask])*c)
    diffraction=ffp.ifft(ffp.fft(hx)*phase_func)
    amplitude_map[k] = sum(np.abs(diffraction)**2)/inputIntensity


x1=x*dx # 実空間の座標

#像の表示
fig,ax=plt.subplots(figsize=(5,4))
ax.plot(x1,amplitude_map)
ax.set_xlabel("x")
ax.set_xticks([-2,-1.5, -1,-0.5, 0,0.5, 1,1.5, 2])
plt.show()

