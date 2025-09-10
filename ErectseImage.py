import numpy as np
from scipy import fftpack as ffp
import matplotlib.pyplot as plt

pairlenz=True # True: 対物レンズと接眼レンズの2枚レンズ系 False: 単レンズ系


def rect(x):
    return np.where(np.abs(x) <= 0.5, 1 , 0)

#パディングを行う関数
def pad(u, m):
    up = np.zeros(len(u)+2*m, dtype=complex)
    up[m:m+len(u)] = u
    return up

#中央部分を切り出す関数
def crop_center(u, N):
    s = (len(u)-N)//2
    return u[s:s+N]

N=4096*16 # サンプリング数
Npad=4096*16 # パディングサイズ
wavelen=532*10**(-6) # 波長(mm)
dx=1e-4 # サンプリング間隔
dz=0.5e-2
print("max x",N*dx/2)
print(dx)
f=1 # レンズの焦点距離
if pairlenz:
    ff=f/2
else:
    ff=f


x=np.linspace(-N/2,N/2-1,N)
xmmpos=0
print(int(xmmpos//dx))
xpos=N//2 +int(xmmpos//dx) #物体点の分布位置 N//2は原点 大体N//2+6000以上にすると光束の分裂が発生
print(xpos)
#物体面の振幅分布
u0=[0.0 for _ in range(len(x))]
u0[xpos]=1
a=2 #1枚目のレンズと物体の距離
b=2 #レンズ間の距離
c=2 #2枚目のレンズと像の距離

distance = np.arange(0, a+b+c + dz, dz) #レンズからの距離

x_mm = (np.arange(N) - N//2) * dx
D1 = 1.5  # 1枚目のレンズの直径(mm)
D2 = 1.5  # 2枚目のレンズの直径(mm)

upad=pad(u0,Npad)
fftu0=ffp.fft(upad)
# ASMによるレンズ前面の波面の計算
nux=ffp.fftfreq(N+2*Npad,dx)
nu_sq=1/wavelen**2-nux**2
mask=nu_sq>0
phase_func=np.zeros(len(nux),dtype=np.complex128)
phase_func[mask]=np.exp(1j*2*np.pi*np.sqrt(nu_sq[mask])*a)
ux=crop_center(ffp.ifft(fftu0*phase_func),N)

amplitude_map = np.zeros((len(distance), N))

#物体面と一枚目のレンズ面の間の波面を各距離で計算
for i in range(len(distance)):
    if distance[i]>a:
        break
    z=distance[i]
    phase_func=np.zeros(len(nux),dtype=np.complex128)
    phase_func[mask]=np.exp(1j*2*np.pi*np.sqrt(nu_sq[mask])*z)
    diffraction=crop_center(ffp.ifft(fftu0*phase_func),N)
    amplitude_map[i, :] = np.abs(diffraction)**2




Px1=rect(x_mm / D1) # 瞳関数
fx1 = np.zeros_like(Px1, dtype=np.complex128)
fx2 = np.zeros_like(Px1, dtype=np.complex128)

inputIntensity=0
for i in range(len(ux)):
    inputIntensity+=np.abs(ux[i])**2*Px1[i]

# レンズの変換を適用
for i in range(N):
    fx1[i]=ux[i]*Px1[i]*np.exp(-1j*np.pi*(x[i]*dx)**2/(wavelen*f))

if pairlenz:
    for i in range(N):
        fx2[i]=fx1[i]*Px1[i]*np.exp(-1j*np.pi*(x[i]*dx)**2/(wavelen*f))
else:
    fx2=fx1
fpad=pad(fx2,Npad)
fftfx=ffp.fft(fpad)

print("one Lenz through")
#一枚目のレンズと二枚目のレンズの間の波面を各距離で計算
for i in range(len(distance)):
    if not a<distance[i]<a+b:
        continue
    z=distance[i]-a
    phase_func=np.zeros(len(nux),dtype=np.complex128)
    phase_func[mask]=np.exp(1j*2*np.pi*np.sqrt(nu_sq[mask])*z)
    diffraction=crop_center(ffp.ifft(fftfx*phase_func),N)
    amplitude_map[i, :] = np.abs(diffraction)**2


# 各距離で角スペクトル法による伝搬計算
phase_func=np.zeros(len(nux),dtype=np.complex128)
phase_func[mask]=np.exp(1j*2*np.pi*np.sqrt(nu_sq[mask])*b)
gx=crop_center(ffp.ifft(fftfx*phase_func),N)

lenzintensity=0
for i in range(len(gx)):
    lenzintensity+=np.abs(gx[i])**2
print("prelenzintensity",lenzintensity)
Px2=rect(x_mm / D2) # 瞳関数
hx1 = np.zeros_like(Px2, dtype=np.complex128)
hx2 = np.zeros_like(Px2, dtype=np.complex128)

# レンズの変換を適用#     if np.max(amplitude_map[i, :]) != 0:
#         amplitude_map[i, :] /= np.max(amplitude_map[i, :])  # 各距離で正規化
for i in range(N):
    hx1[i]=gx[i]*Px2[i]*np.exp(-1j*np.pi*(x[i]*dx)**2/(wavelen*f))

if pairlenz:
    for i in range(N):
        hx2[i]=hx1[i]*Px2[i]*np.exp(-1j*np.pi*(x[i]*dx)**2/(wavelen*f))
else:
    hx2=hx1

lenzintensity=0
for i in range(len(hx2)):
    if Px2[i]==1:
        lenzintensity+=np.abs(hx2[i])**2
print("lenzintensity",lenzintensity)

padhx=pad(hx2,Npad)
ffthx=ffp.fft(padhx)

print("two Lenz through")

#二枚目のレンズと結像面の間の波面を各距離で計算
for i in range(len(distance)):
    if distance[i]<=a+b:
        continue
    z=distance[i]-a-b
    phase_func=np.zeros(len(nux),dtype=np.complex128)
    phase_func[mask]=np.exp(1j*2*np.pi*np.sqrt(nu_sq[mask])*z)
    diffraction=crop_center(ffp.ifft(ffthx*phase_func),N)
    amplitude_map[i, :] = np.abs(diffraction)**2

#結像面での波面を計算
phase_func=np.zeros(len(nux),dtype=np.complex128)
phase_func[mask]=np.exp(1j*2*np.pi*np.sqrt(nu_sq[mask])*c)
imagemap=crop_center(ffp.ifft(ffthx*phase_func),N)
image=np.abs(imagemap)**2
print("Imaged",sum(image)/inputIntensity)
print("Input Intensity:",inputIntensity)
print("Output Intensity:",sum(image))

x1=x*dx # 実空間の座標

#結像面の強度分布をプロット
fig,ax=plt.subplots(figsize=(5,4))
ax.plot(x,image)
ax.set_xlabel("x")
plt.show()

# for i in range(len(distance)):
#     if np.max(amplitude_map[i, :]) != 0:
#         amplitude_map[i, :] /= np.max(amplitude_map[i, :])  # 各距離で正規化

renz1_pos = []
renz2_pos = []
Px1pad=pad(Px1, 1)
Px2pad=pad(Px2, 1)
for i in range(1,len(Px1)-1):
    if Px1[i]==1.0 and Px1[i-1]==0.0:
        renz1_pos.append(x[i-1])
    if Px1[i]==1.0 and Px1[i+1]==0.0:
        renz1_pos.append(x[i-1])
    if Px2[i]==1.0 and Px2[i-1]==0.0:
        renz2_pos.append(x[i-1])
    if Px2[i]==1.0 and Px2[i+1]==0.0:
        renz2_pos.append(x[i-1])

#強度のマップを表示
fig1, ax1 = plt.subplots(figsize=(16, 8))
extent = [distance[0], distance[-1], x1[0], x1[-1]]
I = amplitude_map
I /= I.max()
ax1.plot([a, a], [renz1_pos[0]*dx,renz1_pos[1]*dx], color='blue', linestyle='--', label='1st Lens Position')
ax1.plot([a+b, a+b], [renz2_pos[0]*dx,renz2_pos[1]*dx], color='blue', linestyle='--', label='2nd Lens Position')
im = ax1.imshow(np.log10(I + 1e-12).T, extent=extent, origin='lower', aspect='auto', cmap='hot', vmin=-6, vmax=0)
#im = ax1.imshow(amplitude_map.T, extent=extent, origin='lower', aspect='auto', cmap='gray')  # カラーマップを 'gray' に変更


ax1.set_xlabel("$z$")
ax1.set_ylabel("$x$") 
fig1.colorbar(im, ax=ax1, label="Amplitude Intensity")
fig1.savefig("Erectimage_amplitude_map.png")
plt.show()

