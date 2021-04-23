# %%
import numpy as np
import matplotlib.pyplot as plt

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

original_image = np.zeros((256, 256))
original_image[123:132,123:132] = 1.0
first_pass = median_filtering(original_image, 9)
second_pass = median_filtering(first_pass, 9)
third_pass = median_filtering(second_pass, 9)

# %%
fig, ax = plt.subplots(1,4,figsize=(1 + 3*4.5,4))
display_axis(ax[0], original_image[120:135,120:135], 'Original Image (Cropped)')
display_axis(ax[1], first_pass[120:135,120:135], 'First Pass (Cropped)')
display_axis(ax[2], second_pass[120:135,120:135], 'Second Pass (Cropped)')
display_axis(ax[3], third_pass[120:135,120:135], 'Third Pass (Cropped)')
fig.tight_layout()
fig.savefig('median.pdf', format='pdf', bbox_inches='tight')
# %%
