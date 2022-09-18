fname1 = "ref.exr"
fname2 = "acr.exr"
fname3 = "trt.exr"
fname_Back = ""
output_file = "pdf"

import cv2
import numpy as np

import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from pylab import *


img1 = cv2.imread(fname1, -1)
img2 = cv2.imread(fname2, -1)
img3 = cv2.imread(fname3, -1)

print(img1.shape)
print(img2.shape)
print(img3.shape)

imgAll = np.zeros((img1.shape[0], img1.shape[1], 3))
imgAll[:,:,0] = img1 
imgAll[:,:,1] = img2  
imgAll[:,:,2] = img3  

mode = 0 # BSDF
# mode = 1 # PDF
prct = 99.9


matplotlib.rc("pdf", fonttype=42)
matplotlib.rcParams["font.family"] = "Times New Roman"

cubicL = np.loadtxt("cubicL.txt")
cm = LinearSegmentedColormap.from_list("cubicL", cubicL, N=256)

imgF = np.log1p(imgAll)
if mode == 1 and len(imgF.shape) > 2:
    imgF = np.mean(imgF, 2)
if len(fname_Back) > 0:
    imgB = np.log1p(cv2.imread(fname_Back, -1))
    if mode == 1 and len(imgB.shape) > 2:
        imgB = np.mean(imgB, 2)

imgShape = imgF.shape
if len(fname_Back) > 0:
    assert np.all(imgB.shape == imgShape)

maxVal = np.array([0.1, 0.1])
maxVal[0] = max(maxVal[0], np.percentile(imgF, prct))
if len(fname_Back) > 0:
    maxVal[1] = max(maxVal[1], np.percentile(imgB, prct))

x = linspace(0, 360, 361).astype(int)
x = x*pi/180
y = linspace(0.0, 1.0, 300)

xgrid, ygrid = meshgrid(x, y)

ndim = 3 if mode == 0 else 1
if fname_Back:
    ndim = ndim << 1

zgrid = np.zeros((xgrid.shape[0], xgrid.shape[1], ndim))
for i in range(0, zgrid.shape[0]):
    for j in range(0, zgrid.shape[1]):
        theta = xgrid[i, j]
        rad = ygrid[i, j]

        fx = rad*np.cos(theta)
        fy = rad*np.sin(theta)

        ix = min(int(0.5*(1.0 + fx)*imgShape[1]), imgShape[1] - 1)
        iy = min(int(0.5*(1.0 - fy)*imgShape[0]), imgShape[0] - 1)

        if mode == 0:
            for k in range(0, 3):
                zgrid[i, j, k] = imgF[iy, ix, k]
            if len(fname_Back) > 0:
                for k in range(0, 3):
                    zgrid[i, j, 3 + k] = imgB[iy, ix, k]
        else:
            zgrid[i, j, 0] = imgF[iy, ix]
            if len(fname_Back) > 0:
                zgrid[i, j, 1] = imgB[iy, ix]

titles = ["Groundtruth pdf", "Unbiased pdf", "Approximate pdf", "Red (back)", "Green (back)", "Blue (back)"] if mode == 0 else ["Front", "Back"]
if mode == 0:
    fsize = (26 if len(fname_Back) > 0 else 13, 5)
else:
    fsize = (10 if len(fname_Back) > 0 else 5, 5)

fig = figure(figsize=fsize)
m = 2 if len(fname_Back) > 0 else 1
if mode == 0:
    m = m*3
for i in range(0, m):
    ax = fig.add_subplot(1, m, i + 1,  polar=True)
    pcolormesh(xgrid, ygrid, zgrid[:, :, i], cmap=cm, vmin=0.0, vmax=maxVal[i//3], linewidth=0, rasterized=True)
    yticks([])

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(18)

    cb = colorbar(orientation="horizontal", pad=0.12)
    cb.ax.set_xticklabels(cb.ax.get_xticklabels(), fontsize=18)

    title(titles[i], y=1.15, fontsize=22)

tight_layout()
savefig("%s.pdf" % output_file)
savefig("%s.png" % output_file, dpi=120)

# show()
