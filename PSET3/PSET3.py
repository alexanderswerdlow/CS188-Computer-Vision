# In[ ]:
# Install OpenCV version 4.5.1.48 as it includes SIFT.
# get_ipython().run_line_magic('pip', 'install opencv-contrib-python==4.5.1.48')
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'inline')
import copy
import os
import random
import numpy as np
import scipy
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import display

def display_color(x: np.array, normalized:bool = False):
    plt.figure(figsize=(10,10))
    if not normalized:
        plt.imshow(x,vmin=0,vmax=1)
    else:
        plt.imshow(x/x.max(),vmin=0,vmax=1)
    return plt

def plot_correspondences(image1, image2, correspondences, color):
    image = np.concatenate((image1, image2), axis=1)
    for correspondence in correspondences:
        point1, point2 = correspondence
        point1 = (int(round(point1[0])), int(round(point1[1])))
        point2 = (int(round(point2[0])), int(round(point2[1])))
        cv2.circle(image, point1, 10, color, 2, cv2.LINE_AA)
        cv2.circle(image, tuple([point2[0] + image1.shape[1], point2[1]]), 10, 
                   color, 2, cv2.LINE_AA)
        cv2.line(image, point1, tuple([point2[0] + image1.shape[1], point2[1]]), 
                 color, 2)
    plot = display_color(image)
    return plot


# # Question 3

# ### Image Stitching

# In this question, you will be stitching together two images of the same scene (images assumed to be in left to right order) taken from different camera viewpoints to form a panorama of the scene. This task will require implementing a pipeline with the following steps:
# 
# 1. Extract SIFT keypoints and descriptors from each image and propose possible correspondences by matching SIFT descriptors between the two images. Note that this step outputs some false correspondences, which will be pruned in the next step.
# 2. Estimate the homography between the two images using the following RANSAC loop:
# 
# ```
# For N iterations:
#     i. Get random subset of correspondences.
#     ii. Compute the homography H using homogeneous direct linear transform (DLT) applied to the random subset of correspondences.
#     iii. Count the number of inliers, where inliers are the correspondences (in the whole set of correspondences) that the homography 
#     fits well (using Euclidean distance as the error metric).
#     iv. Keep the homography H with the largest number of inliers and H's corresponding set of inliers.
# ```
# 
# 3. Recompute the homography H using the set of inliers from step 2.
# 4. Use the homographies obtained from step 3 to stitch together the images to form a panorama.

# Load the two images to be stitched together.

# In[ ]:


image1 = np.asarray(Image.open('Data/Problem_3/uttower_left.jpg'))
image2 = np.asarray(Image.open('Data/Problem_3/uttower_right.jpg'))
plot = display_color(np.concatenate((image1, image2), axis=1))

def run_sift(image, num_features):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=num_features)
    return sift.detectAndCompute(gray,None)

def find_sift_correspondences(kp1, des1, kp2, des2, ratio):
    correspondences = []
    for idx, kp in enumerate(kp1):
        distances = ((des2-des1[idx])**2).sum(axis=1)  # compute distances
        ndx = distances.argsort()
        if distances[ndx[0]] < ratio * distances[ndx[1]]:
            correspondences.append((kp.pt, kp2[ndx[0]].pt))

    return correspondences

kp1, des1 = run_sift(image1, 2000)
kp2, des2 = run_sift(image2, 2000)
correspondences = find_sift_correspondences(kp1, des1, kp2, des2, 0.6)
plot = plot_correspondences(image1, image2, correspondences, (0, 0, 255))
os.makedirs('Data/Solutions', exist_ok=True)
plot.savefig('Data/Solutions/question_3_3.pdf', format='pdf', bbox_inches='tight')

def compute_homography(correspondences):
    a = np.zeros((2 * len(correspondences), 9))
    for i,((x1, y1), (x2, y2)) in enumerate(correspondences):
        a[2 * i] = [-x1, -y1, -1, 0, 0, 0, x1*x2, y1*x2, x2]
        a[2 * i + 1] = [0, 0, 0, -x1, -y1, -1, x1*y2, y1*y2, y2]
    u, s, v = np.linalg.svd(a)
    return v[8].reshape((3, 3))

def apply_homography(points, homography):
    a = np.array(points).T # Each column is an x, y
    a = np.vstack([a, np.ones((1, a.shape[1]))]) # Convert to homogeneous
    b = (homography @ a).T # Apply transformation and transpose
    return b[:,:2]/b[:,[-1]] # Convert back to heterogeneous, each row is now (x, y)

def compute_inliers(homography, correspondences, threshold):
    def euclidian_error(correspondence):
        p1 = np.array([[correspondence[0][0]], [correspondence[0][1]],[1]])
        p2 = np.array([[correspondence[1][0]], [correspondence[1][1]]])
        p2_est = np.dot(homography, p1) # Apply homography
        p2_est = p2_est[:-1]/p2_est[-1] # Revert to (x, y)
        return np.linalg.norm(p2 - p2_est) # Calculate euclidean error

    inliers = []
    outliers = []
    for c in correspondences:
        d = euclidian_error(c)
        if d < threshold:
            inliers.append(c)
        else:
            outliers.append(c)

    return inliers, outliers

def ransac(correspondences, num_iterations, num_sampled_points, threshold):
    max_inliers = ([], [], [], 0)
    for i in range(num_iterations):
        selected = random.sample(correspondences, num_sampled_points)
        h = compute_homography(selected)
        inliers, outliers = compute_inliers(h, correspondences, threshold)
        if len(inliers) > max_inliers[3]:
            max_inliers = (h, inliers, outliers, len(inliers))
    return max_inliers[:3]

_, inliers, outliers = ransac(correspondences, 50, 6, 3)
inliers_plot = plot_correspondences(image1, image2, inliers, (0, 255, 0))
inliers_plot.savefig('Data/Solutions/question_3_8_inliers.pdf', format='pdf', bbox_inches='tight')
outliers_plot = plot_correspondences(image1, image2, outliers, (255, 0, 0))
outliers_plot.savefig('Data/Solutions/question_3_8_outliers.pdf', format='pdf', bbox_inches='tight')

def interpolate(image, loc):
    x, y = int(loc[0]), int(loc[1])
    dx, dy = loc[0] - x, loc[1] - y
    a = image[x, y] * (1 - dx)*(1-dy)
    b = image[x + 1, y] * dx * (1 - dy)
    c = image[x, y + 1] * (1 - dx) * dy
    d = image[x + 1, y + 1] * dx * dy
    return a + b + c + d

def stitch_image_given_H(image1, image2, homography):
    h_inv = np.linalg.inv(homography)

    warped = np.zeros((image1.shape[0], image1.shape[1] + image2.shape[1], 3))
    warped[:image1.shape[0],:image1.shape[1]] = image1
    for y in range(warped.shape[0]):
        for x in range(warped.shape[1]):
            p = np.array([x, y, 1])
            p_prime = np.dot(homography, p)
            p_prime = p_prime[:-1]/p_prime[-1]

            if (p_prime[0] < 0 or p_prime[1] < 0
                    or p_prime[0] > image2.shape[1] - 1 or p_prime[1] > image2.shape[0] - 1):
                continue

            warped[y][x] = interpolate(image2, (p_prime[1], p_prime[0]))

    return warped

def stitch_image(image1, image2, num_features, sift_ratio, ransac_iter, ransac_sampled_points, inlier_threshold, use_ransac=True):
    kp1, des1 = run_sift(image1, num_features)
    kp2, des2 = run_sift(image2, num_features)
    correspondences = find_sift_correspondences(kp1, des1, kp2, des2, sift_ratio)
    if use_ransac:
        _, correspondences, _ = ransac(correspondences, ransac_iter, ransac_sampled_points, inlier_threshold)
    h = compute_homography(correspondences)
    return stitch_image_given_H(image1, image2, h)

# The black part is a region of the stitched image that does not come from the two original images.
stitched_image = stitch_image(image1, image2, 2000, 0.6, 50, 6, 3)
stitched_plot = display_color(stitched_image / 255.0)
stitched_plot.savefig('Data/Solutions/question_3_12.pdf', format='pdf', bbox_inches='tight')


# #### **Answer 3.13**

# Execute the cell below and copy the saved image on Overleaf for Question 3.13. The cell runs the image stitching function to stitch the two sample images together (assuming that they are in left to right order) without filtering out outlier correspondences using RANSAC. You should observe that without RANSAC, the stitching does not work properly, showing the importance of RANSAC in the image stitching pipeline. *Note:* This cell may take up to a few minutes to execute.


# The black part is a region of the stitched image that does not come from the two original images.
stitched_image = stitch_image(image1, image2, 2000, 0.6, 50, 6, 3, False)
stitched_plot = display_color(stitched_image / 255.0)
stitched_plot.savefig('Data/Solutions/question_3_13.pdf', format='pdf', bbox_inches='tight')


# # Question 4

# ### Olympic Champion Using Homography

# In this question, you will be making yourself the new World Swimming Champion using homography.

# You are given the following image from the London 2012 Olympics. 

# In[ ]:

img = np.asarray(Image.open('Data/Problem_4/pool-vfx.jpg'))
_ = display_color(img/255.0)
img_final = Image.open('Data/Problem_4/question_4.png')
display(img_final)
name_img = np.asarray(Image.open('Data/Problem_4/flag.png').convert('RGB'))
cp = np.asarray(Image.open('Data/Problem_4/Corresponding_points.png'))
_ = display_color(cp/255.0)

A_1 = (0, 0)
B_1 = (0, 331)
C_1 = (1659, 0)
D_1 = (1659, 331)
correspondence = [
                  ([334,158], A_1),
                  ([340,190], B_1),
                  ([528,157], C_1),
                  ([545,187], D_1),
]


homography = compute_homography(correspondence)


# Now, you will need to stitch the name+flag image into the original Olympic pool image. Complete the function ```stitch_image_given_H_new(pool_image, name_flag_image)``` for this task, which stitches the name+flag image into the original Olympic pool image. Copy paste your solution in the cell below on Overleaf for Question 4.2.
# 
# This function should be similar to the stitching function you wrote previously with minor differences: (1) Previously, since you had to stitch two images side by side, the output image had twice the number of columns as the original image. For this question, since you will be stitching the name+flag image inside the pool image, the output image will have the same number of columns as the input pool image. In other words the output will have the same dimension as `pool_image`. (2) If a pixel location in the stitched image is valid in `pool_image` and has a valid inverse-warped pixel location in `name_flag_image`, then you will use the pixel value from `name_flag_image` instead of averaging both images' pixel locations.

def stitch_image_given_H_new(pool_image, name_flag_image, homography):
    warped = np.copy(pool_image)
    for y in range(warped.shape[0]):
        for x in range(warped.shape[1]):
            p = np.array([x, y, 1])
            p_prime = np.dot(homography, p)
            p_prime = p_prime[:-1]/p_prime[-1]
            if (p_prime[0] < 0 or p_prime[1] < 0 or p_prime[0] > name_flag_image.shape[1] - 1 or p_prime[1] > name_flag_image.shape[0] - 1):
                continue
            warped[y][x] = interpolate(name_flag_image, (p_prime[1], p_prime[0]))

    return warped

new_olympic_champion = stitch_image_given_H_new(img, name_img, homography)
plot = display_color(new_olympic_champion,True)
os.makedirs('Data/Solutions', exist_ok=True)
plot.savefig('Data/Solutions/question_4.png', format='png', bbox_inches='tight')


# # Question 5

# ### Eight-Point Algorithm

# In this question, you will use the eight-point algorithm to reconstruct 3D points associated with some 
# correspondences between two images of the same scene. For this task, you will implement a pipeline with three broad steps:
# 1. Implement the eight-point algorithm to estimate the essential matrix.
# 2. Compute the translation and rotation between the cameras' coordinate frames using the essential matrix.
# 3. Reconstruct the 3D points by solving for their depths. Combining the depths with the 2D points will yield the reconstructed 3D points.

# Load the correspondences. The correspondences are given as a list of tuples $((x_1, y_1), (x_2, y_2))$, where $(x_1, y_1)$ and $(x_2, y_2)$ are the corresponding points from the first and second image, respectively.

# In[ ]:

import numpy as np

def format_correspondences(correspondences):
    formatted_corr = []
    for correspondence in correspondences:
        point1, point2 = correspondence[0:2], correspondence[2:]
        formatted_corr.append((point1, point2))
    return formatted_corr

correspondences = np.load('Data/Problem_5/correspondences.npy')
correspondences = format_correspondences(correspondences)

def compute_essential_matrix(correspondences):
    a = np.zeros((len(correspondences), 9))
    for i,((x1, y1), (x2, y2)) in enumerate(correspondences):
        a[i] = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]
    _, _, v = np.linalg.svd(a)
    f = v[8].reshape((3, 3))
    u, s, v = np.linalg.svd(f)
    s[2] = 0
    f = u @ np.diag(s) @ v
    return f

def compute_translation_rotation(essential_matrix):
    Rz = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    u, s, v = np.linalg.svd(essential_matrix)
    t_hat = (u @ Rz) @ (np.diag(s) @ u.T)
    r = (u @ Rz.T) @ v
    t = np.array([t_hat[2][1], t_hat[0][2], t_hat[1][0]]) # Extract translation from skew symmetric
    return t, r, t_hat

# def compute_essential_matrix(correspondences):
#     A = np.concatenate([np.array([[x0 * x1, x0 * y1, x0, y0 * x1, y0 * y1, y0, x1, y1, 1]]) for (x0, y0), (x1, y1) in correspondences])
#     _, _ , q = np.linalg.svd(A, full_matrices=False)
#     Q_0 = np.reshape(q[-1],(3,3))
#     u, s, vh = np.linalg.svd(Q_0, full_matrices=False)
#     s_diag = np.diag(s)
#     s_diag[-1,-1] = 0
#     return np.matmul(np.matmul(u, s_diag), vh)

# def compute_translation_rotation(essential_matrix):
#     u, s, vh = np.linalg.svd(essential_matrix, full_matrices=False)
#     s_diag = np.diag(s)
#     m_r = np.array([[0,1,0],[-1,0,0],[0,0,1]])
#     r = np.matmul(np.matmul(u, m_r), vh)
#     m_t_hat = np.array([[0, -1,0 ],[1,0,0],[0,0,1]])
#     t_hat = np.matmul(np.matmul(u, m_t_hat), np.matmul(s_diag, u.T))
#     t = np.array([t_hat[2,1], t_hat[0,2], t_hat[1,0]])
#     return t, r, t_hat

essential_matrix = compute_essential_matrix(correspondences)
translation, rotation, t_hat = compute_translation_rotation(essential_matrix)
print("Translation vector: ", translation)
print("Rotation matrix: \n", rotation)
print("T_hat: \n", t_hat)
print("R^T: \n", np.transpose(rotation))
print("R^-1: \n", np.linalg.inv(rotation))

def compute_depths(correspondences, translation, rotation):
    depths = []
    for c in correspondences:
        x1 = np.array([*c[0], 1])
        x2 = np.array([*c[1], 1])
        a = np.array([-rotation @ x1, x2]).T
        depths.append(np.linalg.pinv(a) @ translation)
    return depths

def reconstruct_3d(correspondences, depths):
    points = []
    for c, d in zip(correspondences, depths):
        x1 = np.array([*c[0], 1]) * d[0]
        x2 = np.array([*c[0], 1]) * d[1]
        points.append((x1, x2))
    return points

essential_matrix = compute_essential_matrix(correspondences)
translation, rotation, _ = compute_translation_rotation(essential_matrix)
depths = compute_depths(correspondences, translation, rotation)
corr_3d = reconstruct_3d(correspondences, depths)
rel_errors = []
for (point1_2d, point2_2d), (point1_3d, point2_3d) in zip(correspondences, corr_3d):
    warped_point1_3d = np.matmul(rotation, point1_3d) + translation
    warped_point1_2d = warped_point1_3d / warped_point1_3d[2]
    rel_errors.append(np.linalg.norm((warped_point1_2d[:2] - point2_2d)) / np.linalg.norm(point2_2d))
print(np.mean(rel_errors))
