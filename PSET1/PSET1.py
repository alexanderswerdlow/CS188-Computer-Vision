# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # PSET 1
# %% [markdown]
# These are some of the libraries/modules you will require for this homework.

# %%
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import copy
import os

path = 'Data'

# %% [markdown]
# These are some functions which will be useful throught the homework to (1) display a single grayscale image, (2) display multiple images using subplots, (3) computing the relative absolute distance between two images.

# %%
def display_gray(x: np.array, normalized:bool = False):
    if not normalized:
        plt.imshow(x,cmap='gray',vmin=0,vmax=1)
    else:
        plt.imshow(x/x.max(),cmap='gray',vmin=0,vmax=1)


# %%
def display_axis(ax: plt.axis, x: np.array, title: str, normalized:bool = False):
    if not normalized:
        ax.imshow(x,cmap='gray',vmin=0,vmax=1)
    else:
        ax.imshow(x/x.max(),cmap='gray',vmin=0,vmax=1)
    ax.set_title(title,size=18)


# %%
def rel_l1_dist(x1: np.array, x2: np.array):
    return np.abs(x1-x2).sum()/np.abs(x1).sum()

# %% [markdown]
# Load the house image from `Data/Singles` using Pillow (PIL) into a 2D-array `img_data`. Normalize the image by dividing it with 255. 

# %%
image = Image.open(os.path.join(path,'Singles/house.png'))
img_data = np.asarray(image)/255

# %% [markdown]
# Display the image using `display_gray` defined above.

# %%
display_gray(img_data)

# %% [markdown]
# Print the size of the image

# %%
print(f'Image size: {img_data.shape[0]}x{img_data.shape[1]}')

# %% [markdown]
# # Question 2
# %% [markdown]
# ### 2D-Convolution
# %% [markdown]
# Here you will be implementing the 2D convolution operation using `for` loops. You have to complete the function `conv2D(image, kernel)`.  
# 
# For this homework assume that you are given a grayscale `image` (house) and you want to convolve it with a `kernel` (for example, identity, average or gaussian) such that the output image has the same size as the input `image` (you will need to zero pad the `image` appropriately on all the sides before performing the convolution). The function should return convolution of `image` and `kernel`. 
# 
# *Note:* The origin of the kernel should be the center pixel of the filter, while the origin for the image should be the top left pixel of the image before zero padding. For this homework we will assume that all the filters are `square` and `odd-sized`.
# %% [markdown]
# ### **Answer 2:**
# 
# Copy paste your solution in the cell below on overleaf for Question 2.

# %%
# Write your answer in this cell. Then copy paste the code into the overleaf file corresponding to Question 2.

def conv2D(image: np.array, kernel: np.array = None):
    pad = kernel.shape[0] // 2
    padded_img = np.pad(image, ((pad,pad),(pad,pad)), 'constant')
    new_image = np.empty_like(image)
    for r in range(image.shape[0]):
        for c in range(image.shape[0]):
            new_image[r][c] = np.tensordot(padded_img[r:r + kernel.shape[0], c:c + kernel.shape[0]], kernel)
    return new_image

# %% [markdown]
# One easy way to verify the correctness of the implementation is to make sure that convolving with an identity filter returns the same image. Make sure that you dont get an assertion error.

# %%
def identity_filter(size: int):
    assert size%2 == 1
    iden_filt = np.zeros((size,size))
    iden_filt[size//2,size//2]=1
    return iden_filt


# %%
iden_filt = identity_filter(5)
conv_iden = conv2D(img_data, iden_filt)
assert np.abs(img_data-conv_iden).sum() == 0.0

# %% [markdown]
# # Question 3
# %% [markdown]
# ### Image Blurring and Denoising
# %% [markdown]
# In this question you will be using convolution to perform image blurring and denoising. 
# %% [markdown]
# Average/Box Filter: This is the standard box filter which computes the mean of all the pixels inside the filter.

# %%
def average_filter(size: int):
    assert size%2 == 1
    return 1.0 * np.ones((size,size))/(size**2)

# %% [markdown]
# ## (1) 
# 
# ### Gaussian Filter
# %% [markdown]
# 
# 
# The formula for the distribution of a 2-D isotropic gaussian distribution with variance $\sigma^2$ and mean $= [\mu_x, \mu_y]$ is given by 
# %% [markdown]
# $p(x,y) = \dfrac{1}{2\pi\sigma^2}\exp{\big(-\dfrac{(x-\mu_x)^2+(y-\mu_y)^2}{2\sigma^2}\big)}$
# %% [markdown]
# Using the equation above, complete the function `gaussian_filter(size, sigma)` which given a filter size and standard deviation, returns a centered gaussian filter. Unlike the usual 2-D gaussian which is defined on a continuous space, for the gaussian filter you will assume that $x,y,\mu_x,\mu_y$ are discrete integers.
# 
# *Note:* Don't forget to normalize the filter to make sure that it sums to 1.
# %% [markdown]
# ### **Answer 3.1:**
# Copy paste your solution in the cell below on overleaf for Question 3.1.

# %%
# Write your answer in this cell.

def gaussian_filter(size: int, sigma: float):
    filter = np.zeros((size, size))
    center = size//2
    for i in range(size):
        for j in range(size):
            filter[i][j] = (1/(2*np.pi*sigma**2)) * np.exp(-((j-center)**2 + (i-center)**2)/(2 * sigma**2))
    return filter/filter.sum()

# %% [markdown]
# ## (2)
# %% [markdown]
# ### **Answer 3.2:**
# %% [markdown]
# Execute the cell below and copy the saved image on overleaf for Question 3.2.

# %%
fig, ax = plt.subplots(1,4,figsize=(1 + 4*4.5,4))
for i in range(1,5):
    gauss_filt = gaussian_filter(21,i)
    display_axis(ax[i-1],gauss_filt,f'\u03C3={i}', normalized=True)
fig.tight_layout()
fig.savefig('Data/Solutions/question_3_2.pdf', format='pdf', bbox_inches='tight')

# %% [markdown]
# ### Image Blurring
# %% [markdown]
# In this sub-part you will see that the average and gaussian filter defined above can be used for image blurring. If your implementation of Conv2D and gaussian filter is correct then you should observe that increasing the filter size for the average filter and the filter size/variance for the gaussian filter will increase the blurring. So the images on the right will be more blurred. 
# %% [markdown]
# #### Average Filter

# %%
fig, ax = plt.subplots(1,4,figsize=(1 + 4*4.5,4))
for i in range(1,5):
    size = 4*i+1
    avg_filt = average_filter(size)
    conv_avg = conv2D(img_data, avg_filt)
    display_axis(ax[i-1],conv_avg,f'Kernel Size={size}')

# %% [markdown]
# ## (3)
# #### Gaussian Filter
# %% [markdown]
# ### **Answer 3.3:**
# %% [markdown]
# Execute the cell below and copy the saved image on overleaf for Question 3.3.

# %%
fig, ax = plt.subplots(2,4,figsize=(1 + 4*4.5,2*4))
for i in range(1,5):
    sigma = 4*(i-1)+1
    s = 4*i + 1
    gauss_filt = gaussian_filter(21,sigma)
    conv_gauss = conv2D(img_data, gauss_filt)
    display_axis(ax[0,i-1],conv_gauss,f'\u03C3={sigma}')
    gauss_filt = gaussian_filter(s,5)
    conv_gauss = conv2D(img_data, gauss_filt)
    display_axis(ax[1,i-1],conv_gauss,f'Kernel Size={s}')
fig.tight_layout()
fig.savefig('Data/Solutions/question_3_3.pdf', format='pdf', bbox_inches='tight')

# %% [markdown]
# ## Image Denoising
# %% [markdown]
# In this question you will use `conv2D` to perform image denoising. You will use three types of filtering for denoising: (i) average, (ii) gaussian and (iii) median. Average and Gaussian filtering can easily be performed using the current implementation of the `conv2D` function. However, the median filter cannot be characterized using a known filter unlike average, gaussian. You will write a function for performing median filtering.
# 
# You will perform all the three types of filtering and report best filtering method (you may find the `rel_abs_dist` function useful for this part.
# %% [markdown]
# Display the noisy image

# %%
noisy_img_data = np.asarray(Image.open('Data/Singles/noisy_house.png'))
noisy_img_data = noisy_img_data/255
display_gray(noisy_img_data)

# %% [markdown]
# ## (5) 
# ### Median filtering
# %% [markdown]
# Complete the function `median_filtering(image, kernel_size)` which takes the `image` as input along with the `kernel_size` and returns the median filtered output which has the same size as the input image (you need to perform zero padding).
# %% [markdown]
# ### **Answer 3.5:**
# Copy paste your solution in the cell below on overleaf for Question 3.5.

# %%
# Write your answer in this cell. Then copy paste the code into the overleaf file corresponding to Question 3 (d).

def median_filtering(image: np.array, kernel_size: int = None):
    def sanitize_bounds(edge: int):
        if edge < 0:
            return 0
        elif edge >= image.shape[0]:
            return image.shape[0] - 1
        else:
            return edge

    pad = kernel_size // 2
    new_image = np.empty_like(image)
    for r in range(image.shape[0]):
        for c in range(image.shape[0]):
            bottom_r = sanitize_bounds(r - pad)
            top_r = sanitize_bounds(r + pad + 1)
            left_c = sanitize_bounds(c - pad)
            right_c = sanitize_bounds(c + pad + 1)
            new_image[r][c] = np.median(image[bottom_r : top_r, left_c : right_c])
    return new_image

# %% [markdown]
# Perform the 3 types of filtering.

# %%
avg_filt = average_filter(7)
gauss_filt = gaussian_filter(7,3)
avg_filt_noisy_img = conv2D(noisy_img_data, avg_filt)
gauss_filt_noisy_img = conv2D(noisy_img_data, gauss_filt)
median_filt_noisy_img = median_filtering(noisy_img_data,7)

# %% [markdown]
# Display all the images.
# %% [markdown]
# ### **Answer 3.6:**
# %% [markdown]
# Execute the cell below and copy the saved image on overleaf for Question 3.6.

# %%
fig, ax = plt.subplots(1,5,figsize=(1 + 5*4.5,4))
display_axis(ax[0],noisy_img_data, 'Noisy Image')
display_axis(ax[1],img_data, 'Noiseless Image')
display_axis(ax[2],avg_filt_noisy_img,'Average Filtering')
display_axis(ax[3],gauss_filt_noisy_img,'Gaussian Filtering')
display_axis(ax[4],median_filt_noisy_img,'Median Filtering')
fig.tight_layout()
fig.savefig('Data/Solutions/question_3_6.pdf', format='pdf', bbox_inches='tight')

# %% [markdown]
# Relative absolute distance

# %%
print(f'Average Filtering: {rel_l1_dist(img_data, avg_filt_noisy_img)}')
print(f'Gaussian Filtering: {rel_l1_dist(img_data, gauss_filt_noisy_img)}')
print(f'Median Filtering: {rel_l1_dist(img_data, median_filt_noisy_img)}')

# %% [markdown]
# ## Question 4 
# %% [markdown]
# ## Gradients
# %% [markdown]
# In this question you will be using convolution, `conv2D` to compute image gradients. Gradients are useful in obtaining the edges in an image. Multiple edge level features can be combined to obtain higher level features which are useful in image classification. 
# 
# Design a filter to compute the gradient along the horizontal and vertical direction. After convolving with a filter which computes the gradient along the horizontal direction you should observe that all the vertical edges in the filtered image and vice-versa. 
# 
# *Hint:* See Prewitt filter
# %% [markdown]
# ## (1)
# 
# Design a filter `gradient_x` for computing the horizontal gradient (along the x direction) using `conv2D`. 
# %% [markdown]
# ### **Answer 4.1:**
# Copy paste your solution in the cell below on overleaf for Question 4.1.

# %%
gradient_x = (1/6) * np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]])

# %% [markdown]
# ## (2)
# 
# Design a filter `gradient_y` for computing the vertical gradient (along the y direction) using `conv2D`.
# %% [markdown]
# ### **Answer 4.2:**
# Copy paste your solution in the cell below on overleaf for Question 4.2.

# %%
gradient_y = (1/6) * np.array([[-1, -1, -1],[0, 0, 0],[1, 1, 1]])

# %% [markdown]
# Display the absolute gradient along the horizontal, vertical directions and their sum. You should observe that the gradient in the horizontal (x-direction) is unable to capture the horizontal parts of the rooftops, while the vertical gradient is unable to captures features like the edges of chimney. 
# %% [markdown]
# ### **Answer 4.3:**
# %% [markdown]
# Execute the cell below and copy the saved image on overleaf for Question 4.3.

# %%
fig, ax = plt.subplots(1,3,figsize=(1 + 3*4.5,4))
img_gradient_x = conv2D(img_data, gradient_x)
img_gradient_y = conv2D(img_data, gradient_y)
display_axis(ax[0], np.abs(img_gradient_x), 'Gradient-X')
display_axis(ax[1], np.abs(img_gradient_y), 'Gradient-Y')
display_axis(ax[2], np.abs(img_gradient_x) + np.abs(img_gradient_y), 'Gradient-Sum')
fig.tight_layout()
fig.savefig('Data/Solutions/question_4_3.pdf', format='pdf', bbox_inches='tight')

# %% [markdown]
# ## Question 5
# %% [markdown]
# ### Image Filtering
# %% [markdown]
# In this question you will be completing a function `filtering_2(image, kernel, sigma_int, norm_fac)` which takes as input the `image`, a gaussian filter `kernel` which is the filter for the spatial dimension, `sigma_int` is the standard deviation of the gaussian along the intensity/pixel dimension and `norm_fac` is the normalization factor.
# 
# *Note:* For this filter you have two types of gaussian filters (one along the spatial dimension and the other along the pixel dimension). The gaussian filter along the spatial dimension is 2-D can be obtained using the `gaussian_filter` function you wrote previously, however, the gaussian filter along the intensity dimension is 1-D and non-linear filter. 
# %% [markdown]
# ### **Answer 5.5:**
# Copy paste your solution in the cell below on overleaf for Question 5.5.

# %%
# Write your code in this cell.

def filtering_2(image: np.array, kernel: np.array = None, sigma_int: float = None, norm_fac: float = None):
    pad = kernel.shape[0] // 2
    new_image = np.zeros_like(image)
    for r in range(image.shape[0]):
        for c in range(image.shape[0]):
            norm = 0
            for i in range(r - pad, r + pad + 1):
                for j in range(c - pad, c + pad + 1):
                    if i >= image.shape[0] or j >= image.shape[0] or i < 0 or j < 0:
                        continue
                    intensity_diff = np.abs(image[r][c] - image[i][j])
                    intensity_gaussian = ((1/(np.sqrt(2*np.pi)*sigma_int)) * np.exp(-(intensity_diff**2)/(2 * sigma_int**2))) 
                    spatial_gaussian = kernel[i - (r - pad)][j - (c - pad)]
                    new_image[r][c] += spatial_gaussian * intensity_gaussian * image[i][j]
                    norm += spatial_gaussian * intensity_gaussian
            new_image[r][c] /= norm
    return new_image * norm_fac


# %%
gauss_filt = gaussian_filter(11,3)
gauss_filt_img_data = conv2D(img_data, gauss_filt)
filt_2_img_data = filtering_2(img_data, gauss_filt, sigma_int=0.2, norm_fac=1.0) #These are some reference values, feel free to tune the values.

# %% [markdown]
# Comparison of the new filter with the gaussian filter should show that the new filter preserves edges while smoothing the remaining image.
# %% [markdown]
# ### **Answer 5.6:**
# %% [markdown]
# Execute the cell below and copy the saved image on overleaf for Question 5.6.

# %%
fig, ax = plt.subplots(1,3,figsize=(1 + 3*4.5,4))
display_axis(ax[0], img_data, 'Original Image')
display_axis(ax[1], gauss_filt_img_data, 'Gaussian Filtering')
display_axis(ax[2], filt_2_img_data, 'Filtering 2')
fig.tight_layout()
fig.savefig('Data/Solutions/question_5_6.pdf', format='pdf', bbox_inches='tight')


# %%



