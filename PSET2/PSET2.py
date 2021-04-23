# %%
from IPython import get_ipython

# %%
import numpy as np
import scipy
from PIL import Image
import skimage
from skimage import data
from skimage.transform import warp, AffineTransform
import matplotlib.pyplot as plt
import copy
import os

# %%
def display_gray(x: np.array, normalized: bool = False):
    plt.figure(figsize=(10, 10))
    if not normalized:
        plt.imshow(x, cmap='gray', vmin=0, vmax=1)
    else:
        plt.imshow(x / x.max(), cmap='gray', vmin=0, vmax=1)


# %%
def display_axis(ax: plt.axis, x: np.array, title: str, normalized: bool = False):
    if not normalized:
        ax.imshow(x, cmap='gray', vmin=0, vmax=1)
    else:
        ax.imshow(x / x.max(), cmap='gray', vmin=0, vmax=1)
    ax.set_title(title, size=18)


# %%
def display_axis_bw(ax: plt.axis, x: np.array, title: str, normalized: bool = False, fig=None):
    if not normalized:
        imax = ax.imshow(x, cmap='gray', vmin=0, vmax=1)
    else:
        imax = ax.imshow(x / x.max(), cmap='gray', vmin=0, vmax=1)
    ax.set_title(title, size=18)
    fig.colorbar(imax, ax=ax)


# %%
def display_axis_color(ax: plt.axis, x: np.array, title: str, normalized: bool = False, fig=None):
    if not normalized:
        imax = ax.imshow(x, cmap='coolwarm', vmin=0, vmax=1)
    else:
        imax = ax.imshow(x / x.max(), cmap='coolwarm', vmin=0, vmax=1)
    ax.set_title(title, size=18)
    fig.colorbar(imax, ax=ax)

# %%
# Copy paste your conv2D function from the previous homework here.
def conv2D(image: np.array, kernel: np.array = None):
    pad = kernel.shape[0] // 2
    padded_img = np.pad(image, ((pad, pad), (pad, pad)), 'constant')
    new_image = np.empty_like(image)
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            new_image[r, c] = np.tensordot(padded_img[r:r + kernel.shape[0], c:c + kernel.shape[0]], kernel)
    return new_image



# %%
def make_circle(img: np.array, x: int, y: int, radius: int):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if np.sqrt((x - i) ** 2 + (y - j) ** 2) < 1.0 * radius:
                img[i, j] = 1
    return img


# %%
def draw_circle(img, y, radius):
    rad_buffer = radius + 5
    start = rad_buffer + 5
    end = 255 - rad_buffer - 5
    centers = np.linspace(start, end, int((end - start) / (2 * rad_buffer)))
    for c in centers:
        c = int(c)
        make_circle(img, y, c, radius)
    return img


# %%
blob_img = np.zeros((256, 256))
blob_img = draw_circle(blob_img, 25, 5)
blob_img = draw_circle(blob_img, 50, 10)
blob_img = draw_circle(blob_img, 85, 15)
blob_img = draw_circle(blob_img, 130, 20)
blob_img = draw_circle(blob_img, 200, 35)

# %%
display_gray(blob_img)

# %%
img = copy.deepcopy(blob_img)

# %% 



# def log_filter(size: int, sigma: float):
#     min_x = -(size // 2)
#     idx_range = slice(min_x, min_x+size)
#     indices = np.mgrid[idx_range, idx_range]
#     dist_squared = indices[0]**2 + indices[1]**2
#     lap_of_gaus = (dist_squared - 2*sigma**2)/(2*np.pi*sigma**4) * np.exp(-dist_squared / (2 * sigma**2))
#     return lap_of_gaus

def log_filter(size: int, sigma: float):
    filter = np.zeros((size, size))
    center = size // 2
    for i in range(size):
        for j in range(size):
            x, y = j - center, i - center
            a = (1 - ((x**2 + y**2) / (2 * sigma**2)))
            b = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            filter[i, j] = a * b
    return (sigma**2) * -(1 / (np.pi * sigma ** 4)) * filter

import time
start = time.perf_counter()
sigma_1 = 5/np.sqrt(2)
log_1 = log_filter(21, sigma_1)
sigma_2 = 10/np.sqrt(2)
log_2 = log_filter(31, sigma_2)
sigma_3 = 15/np.sqrt(2)
log_3 = log_filter(41, sigma_3)
sigma_4 = 20/np.sqrt(2)
log_4 = log_filter(51, sigma_4)
sigma_5 = 35/np.sqrt(2)
log_5 = log_filter(81, sigma_5)
print(f"Time: {time.perf_counter() - start}")

fig, ax = plt.subplots(1, 5, figsize=(1 + 5 * 4.5, 4))
display_axis_bw(ax[0], log_1, 'Sigma1', normalized=True, fig=fig)
display_axis_bw(ax[1], log_2, 'Sigma2', normalized=True, fig=fig)
display_axis_bw(ax[2], log_3, 'Sigma3', normalized=True, fig=fig)
display_axis_bw(ax[3], log_4, 'Sigma4', normalized=True, fig=fig)
display_axis_bw(ax[4], log_5, 'Sigma5', normalized=True, fig=fig)
fig.tight_layout()
os.makedirs('Data/Solutions', exist_ok=True)
fig.savefig('Data/Solutions/question_2_7.pdf', format='pdf', bbox_inches='tight')
plt.show()

# from scipy import ndimage

# def identity_filter(size: int):
#     assert size%2 == 1
#     iden_filt = np.zeros((size,size))
#     iden_filt[size//2,size//2]=1
#     return iden_filt

# fig, ax = plt.subplots(1, 5, figsize=(1 + 5 * 4.5, 4))
# display_axis_bw(ax[0], scipy.ndimage.gaussian_laplace(identity_filter(21), sigma=sigma_1), 'Sigma1', normalized=True, fig=fig)
# display_axis_bw(ax[1], scipy.ndimage.gaussian_laplace(identity_filter(31), sigma=sigma_2), 'Sigma2', normalized=True, fig=fig)
# display_axis_bw(ax[2], scipy.ndimage.gaussian_laplace(identity_filter(41), sigma=sigma_3), 'Sigma3', normalized=True, fig=fig)
# display_axis_bw(ax[3], scipy.ndimage.gaussian_laplace(identity_filter(51), sigma=sigma_4), 'Sigma4', normalized=True, fig=fig)
# display_axis_bw(ax[4], scipy.ndimage.gaussian_laplace(identity_filter(81), sigma=sigma_5), 'Sigma5', normalized=True, fig=fig)
# fig.tight_layout()
# os.makedirs('Data/Solutions', exist_ok=True)
# fig.savefig('Data/Solutions/question_2_7.pdf', format='pdf', bbox_inches='tight')
# plt.show()

# %%

import cv2
import time
import numpy as np
import math

points = [(0, 1, 21, 5/math.sqrt(2)), (0, 2, 31, 10/math.sqrt(2)), (1, 0, 41, 15/math.sqrt(2)), (1,1, 51, 20/math.sqrt(2)), (1,2, 81, 35/math.sqrt(2))]

start = time.perf_counter()
fig, ax = plt.subplots(2, 3, figsize=(1 + 3 * 6, 2 * 6))
display_axis_bw(ax[0, 0], img, 'Original Image', fig=fig)
for ind, (i, j, k, l) in enumerate(points):
    display_axis_color(ax[i, j], conv2D(img, -log_filter(k, l)), f"Sigma{ind+1}", fig=fig)
fig.tight_layout()
plt.show()
fig.savefig('Data/Solutions/question_2_8.pdf', format='pdf', bbox_inches='tight')
print(f"Delta Time: {time.perf_counter() - start}")

# start = time.perf_counter()
# fig, ax = plt.subplots(2, 3, figsize=(1 + 3 * 6, 2 * 6))
# display_axis_bw(ax[0, 0], img, 'Original Image', fig=fig)
# for ind, (i, j, k, l) in enumerate(points):
#     display_axis_color(ax[i, j], l**2 * -ndimage.gaussian_laplace(img, l), f"Sigma{ind+1}", fig=fig)
# fig.tight_layout()
# plt.show()
# fig.savefig('Data/Solutions/question_2_8.pdf', format='pdf', bbox_inches='tight')
# print(f"Delta Time: {time.perf_counter() - start}")

# start = time.perf_counter()
# fig, ax = plt.subplots(2, 3, figsize=(1 + 3 * 6, 2 * 6))
# display_axis_bw(ax[0, 0], img, 'Original Image', fig=fig)
# for ind, (i, j, k, l) in enumerate(points):
#     display_axis_color(ax[i, j], -cv2.Laplacian(cv2.GaussianBlur(img,(k, k),l),cv2.CV_64F, scale=l**2), f"Sigma{ind+1}", fig=fig)
# fig.tight_layout()
# plt.show()
# print(f"Delta Time: {time.perf_counter() - start}")

# %%
# Sheared checkerboard
tform = AffineTransform(scale=(1.3, 1.1), rotation=1, shear=0.7,
                        translation=(110, 30))
image = warp(data.checkerboard()[:90, :90], tform.inverse,
             output_shape=(200, 310))

# Two squares
image[30:80, 200:250] = 1
image[80:130, 250:300] = 1

display_gray(image)

# %%0
sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

from skimage import filters
def compute_image_gradient(image: np.array):
    return conv2D(image, sobel_x), conv2D(image, sobel_y)

fig, ax = plt.subplots(1, 3, figsize=(1 + 3 * 4.5, 4))
img_gradient_x, img_gradient_y = filters.sobel_v(image), filters.sobel_h(image)
display_axis(ax[0], np.abs(img_gradient_x), 'Gradient-X')
display_axis(ax[1], np.abs(img_gradient_y), 'Gradient-Y')
display_axis(ax[2], np.abs(img_gradient_x) + np.abs(img_gradient_y), 'Gradient-Sum')
fig.tight_layout()
os.makedirs('Data/Solutions', exist_ok=True)
fig.savefig('Data/Solutions/question_3_2.pdf', format='pdf', bbox_inches='tight')


# fig, ax = plt.subplots(1, 3, figsize=(1 + 3 * 4.5, 4))
# img_gradient_x, img_gradient_y = reversed(np.gradient(image))
# display_axis(ax[0], np.abs(img_gradient_x), 'Gradient-X')
# display_axis(ax[1], np.abs(img_gradient_y), 'Gradient-Y')
# display_axis(ax[2], np.abs(img_gradient_x) + np.abs(img_gradient_y), 'Gradient-Sum')
# fig.tight_layout()
# os.makedirs('Data/Solutions', exist_ok=True)
# fig.savefig('Data/Solutions/question_3_2.pdf', format='pdf', bbox_inches='tight')


# fig, ax = plt.subplots(1, 3, figsize=(1 + 3 * 4.5, 4))
# img_gradient_x, img_gradient_y = conv2D(image, sobel_x), conv2D(image, sobel_y)
# display_axis(ax[0], np.abs(img_gradient_x), 'Gradient-X')
# display_axis(ax[1], np.abs(img_gradient_y), 'Gradient-Y')
# display_axis(ax[2], np.abs(img_gradient_x) + np.abs(img_gradient_y), 'Gradient-Sum')
# fig.tight_layout()
# os.makedirs('Data/Solutions', exist_ok=True)
# fig.savefig('Data/Solutions/question_3_2.pdf', format='pdf', bbox_inches='tight')

# %%
# This is the standard box filter which computes the mean of all the pixels inside the filter.
import stackprinter
stackprinter.set_excepthook(style='darkbg2')

def average_filter(size: int):
    assert size % 2 == 1
    return 1.0 * np.ones((size, size)) / (size ** 2)
    
def grad_covariance(image: np.array, size: int):
    dx, dy = compute_image_gradient(image)
    Ixx, Ixy, Iyy = dx**2, dy*dx, dy**2
    i_xx, i_yy, i_xy = (conv2D(I, average_filter(size)) for I in (Ixx, Iyy, Ixy))
    return (i_xx, i_xy, i_yy)

def harris_response(image: np.array, k: float, size: int):
    i_xx, i_xy, i_yy = grad_covariance(image, size)
    ret = np.empty_like(image)
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            a = np.array([[i_xx[r][c], i_xy[r][c]], [i_xy[r][c], i_yy[r][c]]])
            ret[r, c] = np.linalg.det(a) - k * (np.trace(a)) ** 2
    return ret

coords = np.argwhere(harris_response(image, 0.05, 3) > 0.02)
fig, ax = plt.subplots()
ax.imshow(image, cmap=plt.cm.gray)
ax.plot(coords[:, 1], coords[:, 0], color='cyan', marker='o',
        linestyle='None', markersize=6)
ax.axis((0, 310, 200, 0))
fig.tight_layout()
plt.show()
fig.savefig('Data/Solutions/question_3_5.pdf', format='pdf', bbox_inches='tight')

# from skimage.feature import corner_harris, corner_peaks
# def harris_response(image: np.array, k: float, size: int):
#     return corner_harris(image, k=0.05)

# from skimage.feature import corner_harris, corner_peaks
# coords = np.argwhere(harris_response(image, 0.05, 3) > 0.02)
# fig, ax = plt.subplots()
# ax.imshow(image, cmap=plt.cm.gray)
# ax.plot(coords[:, 1], coords[:, 0], color='cyan', marker='o',
#         linestyle='None', markersize=6)
# ax.axis((0, 310, 200, 0))
# fig.tight_layout()
# plt.show()
# fig.savefig('Data/Solutions/question_3_5.pdf', format='pdf', bbox_inches='tight')

def threshold_harris_response(harris_response: np.array, threshold: float):
    return np.argwhere(harris_response > threshold)

def sort_detections(candidate_detections: np.array, harris_response: np.array):
    sorted_detection_responses = np.argsort(-harris_response[candidate_detections[:,0], candidate_detections[:, 1]])
    return [candidate_detections[ind] for ind in sorted_detection_responses]

def l2_distance(p1: np.array, p2: np.array):
    return np.linalg.norm(p1 - p2, ord=2)

def local_max(sorted_detections: np.array, distance: float):
    valid_detections = []
    while sorted_detections:
        top = sorted_detections.pop(0)
        if not any(l2_distance(top, d) < distance for d in valid_detections):
            valid_detections.append(top)
    return np.asarray(valid_detections)

def non_max_suppression(harris_response: np.array, distance: float, threshold: float):
    t_responses = threshold_harris_response(harris_response, threshold)
    sorted_candidates = sort_detections(t_responses, harris_response)
    return local_max(sorted_candidates, distance)

coords = non_max_suppression(harris_response(image, 0.05, 3), 10.0, 0.02)
fig, ax = plt.subplots()
ax.imshow(image, cmap=plt.cm.gray)
ax.plot(coords[:, 1], coords[:, 0], color='cyan', marker='o',
        linestyle='None', markersize=6)
ax.axis((0, 310, 200, 0))
fig.tight_layout()
plt.show()
fig.savefig('Data/Solutions/question_3_10.pdf', format='pdf', bbox_inches='tight')
# %%