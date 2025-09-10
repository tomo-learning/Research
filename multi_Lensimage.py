import numpy as np
from scipy import fftpack as ffp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
def rect(x_mm, diameter_mm, center_mm):
    """一次元レンズ開口（矩形）関数
    x_mm: mm単位の座標配列
    diameter_mm: レンズ直径 [mm]
    center_mm: レンズ中心位置 [mm]
    """
    half = diameter_mm * 0.5
    return np.where(np.abs(x_mm - center_mm) <= half, 1.0, 0.0)

#パディングを行う関数
def pad(u, m):
    up = np.zeros(len(u)+2*m, dtype=complex)
    up[m:m+len(u)] = u
    return up

#中央部分を切り出す関数
def crop_center(u, N):
    s = (len(u)-N)//2
    return u[s:s+N]

# レンズアレイを波面図に描画する関数
def draw_lensarray_lines(ax, z_plane, Lensarray, x_mm,
                         *, line_color="blue", line_w=2.0, line_alpha=0.9,
                         endpoint_color="white", endpoint_size=10):
    """
    ax            : 描画先 Axes（例: ax1）
    z_plane       : 描画する z 位置（例: a）
    Lensarray     : 形状 [LensNum, N] の 0/1 配列（各 i の開口マスク）
    x_mm          : 縦軸の座標配列 [mm]（N 要素）
    line_color    : 線色（開口区間）
    line_w        : 線幅
    line_alpha    : 線の不透明度
    endpoint_color: 端点の色
    endpoint_size : 端点マーカーのサイズ
    """
    Lensarray = np.asarray(Lensarray)
    for i in range(Lensarray.shape[0]):
        m = np.asarray(Lensarray[i] == 1, dtype=bool)
        if m.ndim != 1:
            raise ValueError("Lensarray の各行は 1 次元配列である必要があります。")

        # 開口(=True) の連続区間の開始/終了インデックスを抽出
        diff = np.diff(m.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends   = np.where(diff == -1)[0] + 1
        if m[0]:
            starts = np.r_[0, starts]
        if m[-1]:
            ends = np.r_[ends, m.size]

        # 各区間を「青い縦線＋白い端点」で描画
        for s, e in zip(starts, ends):
            y0 = x_mm[s]
            y1 = x_mm[e-1]
            if y1 < y0:
                y0, y1 = y1, y0

            # 縦線（z は一定、x が y 軸）
            ax.plot([z_plane, z_plane], [y0, y1],
                    color=line_color, linewidth=line_w, alpha=line_alpha,
                    solid_capstyle="butt", zorder=4)

            # 端点（白）
            ax.plot([z_plane], [y0], marker="o", markersize=endpoint_size,
                    markerfacecolor=endpoint_color, markeredgecolor=endpoint_color,
                    zorder=5, clip_on=False)
            ax.plot([z_plane], [y1], marker="o", markersize=endpoint_size,
                    markerfacecolor=endpoint_color, markeredgecolor=endpoint_color,
                    zorder=5, clip_on=False)

N=4096*32 # サンプリング数
Npad=4096*16 # パディングサイズ
wavelen=532*10**(-6) # 波長(mm)
dx=1e-4 # サンプリング間隔
dz=0.5e-2
f=1 # レンズの焦点距離



x=np.linspace(-N/2,N/2-1,N)
xmmpos=0 #物体点の位置(mm)
print(int(xmmpos//dx))
xpos=N//2 +int(xmmpos//dx) #物体点の分布位置 N//2は原点
print(xpos)
#物体面の振幅分布
u0=[0.0 for _ in range(len(x))]
u0[xpos]=1
a=2 #1枚目のレンズと物体の距離
b=4/3 #レンズ間の距離
c=2 #2枚目のレンズと像の距離

distance = np.arange(0, a+b+c + dz, dz) #レンズからの距離

LensNum=5 # レンズの枚数
x_mm = (np.arange(N) - N//2) * dx
D = 1.5

Lensarray = np.zeros((LensNum, N)) # レンズアレイの配列
Lenscenterarray = np.zeros(LensNum) # レンズ中心位置の配列

# レンズアレイの生成
lenscenter=0
for i in range(LensNum):
    Lensarray[i] = rect(x_mm , D , lenscenter)
    Lenscenterarray[i] = lenscenter
    if lenscenter==0:
        lenscenter+=D
    elif lenscenter>0:
        lenscenter=-lenscenter
    else:
        lenscenter=-lenscenter+D



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

#物体面と一組目のレンズ面の間の波面を各距離で計算
for i in range(len(distance)):
    if distance[i]>a:
        break
    z=distance[i]
    phase_func=np.zeros(len(nux),dtype=np.complex128)
    phase_func[mask]=np.exp(1j*2*np.pi*np.sqrt(nu_sq[mask])*z)
    diffraction=crop_center(ffp.ifft(fftu0*phase_func),N)
    amplitude_map[i, :] = np.abs(diffraction)**2


fx1 = np.zeros_like(u0, dtype=np.complex128)
fx2 = np.zeros_like(u0, dtype=np.complex128)
# レンズの変換を適用
for i in range(LensNum):
    P=Lensarray[i]
    xcenter=Lenscenterarray[i]
    for j in range(N):
        if P[j]==1:
            fx1[j]=ux[j]*np.exp(-1j*np.pi*(x[j]*dx-xcenter)**2/(wavelen*f))
            fx2[j]=fx1[j]*np.exp(-1j*np.pi*(x[j]*dx-xcenter)**2/(wavelen*f))

#一組目のレンズと二組目のレンズの間の波面を各距離で計算
for j in range(LensNum):
    P=Lensarray[j]
    fx=fx2*P
    fpad=pad(fx,Npad)
    fftfx=ffp.fft(fpad)
    
    for i in range(len(distance)):
        if not a<distance[i]<a+b:
            continue
        z=distance[i]-a
        phase_func=np.zeros(len(nux),dtype=np.complex128)
        phase_func[mask]=np.exp(1j*2*np.pi*np.sqrt(nu_sq[mask])*z)
        diffraction=crop_center(ffp.ifft(fftfx*phase_func),N)
        amplitude_map[i, :] += P*(np.abs(diffraction)**2)

gx = np.zeros_like(u0, dtype=np.complex128)
#二組目のレンズ前面の波面計算
for i in range(LensNum):
    P=Lensarray[i]
    fx=fx2*P
    fpad=pad(fx,Npad)
    fftfx=ffp.fft(fpad)
    phase_func=np.zeros(len(nux),dtype=np.complex128)
    phase_func[mask]=np.exp(1j*2*np.pi*np.sqrt(nu_sq[mask])*b)
    gx+=crop_center(ffp.ifft(fftfx*phase_func),N)*P

hx1 = np.zeros_like(gx, dtype=np.complex128)
hx2 = np.zeros_like(gx, dtype=np.complex128)
# レンズの変換を適用
for i in range(LensNum):
    P=Lensarray[i]
    xcenter=Lenscenterarray[i]
    for j in range(N):
        if P[j]==1:
            hx1[j]=gx[j]*np.exp(-1j*np.pi*(x[j]*dx-xcenter)**2/(wavelen*f))
            hx2[j]=hx1[j]*np.exp(-1j*np.pi*(x[j]*dx-xcenter)**2/(wavelen*f))


padhx=pad(hx2,Npad)
ffthx=ffp.fft(padhx)

#二組目のレンズと結像面の間の波面を各距離で計算
for i in range(len(distance)):
    if distance[i]<a+b:
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

x1=x*dx # 実空間の座標

#結像面の強度分布をプロット
fig,ax=plt.subplots(figsize=(5,4))
ax.plot(x,image)
ax.set_xlabel("x")
plt.show()


#強度のマップを表示
fig1, ax1 = plt.subplots(figsize=(16, 8))
extent = [distance[0], distance[-1], x1[0], x1[-1]]
I = amplitude_map
I /= I.max()
im = ax1.imshow(np.log10(I + 1e-12).T, extent=extent, origin='lower', aspect='auto', cmap='hot', vmin=-6, vmax=0)
draw_lensarray_lines(ax1, a, Lensarray, x1,
                     line_color="blue", line_w=2.0, line_alpha=0.95,
                     endpoint_color="white", endpoint_size=2)
draw_lensarray_lines(ax1, a+b, Lensarray, x1,
                     line_color="blue", line_w=2.0, line_alpha=0.95,
                     endpoint_color="white", endpoint_size=2)
ax1.set_xlabel("$z$")
ax1.set_ylabel("$x$") 
fig1.colorbar(im, ax=ax1, label="Amplitude Intensity")
fig1.savefig("Erectimage_amplitude_map.png")


plt.show()

