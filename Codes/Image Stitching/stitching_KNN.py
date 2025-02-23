import cv2
import numpy as np
import random
import os
from datetime import datetime

def create_session_directories():
    """
    Create a new session directory with subdirectories for storing visualizations.
    
    :return: Dictionary containing paths for different visualization types
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join('results', f"session_{timestamp}")
    
    # Create subdirectories
    subdirs = ['keypoints', 'matches', 'transformed', 'final']
    paths = {'base': session_dir}
    
    for subdir in subdirs:
        dir_path = os.path.join(session_dir, subdir)
        os.makedirs(dir_path, exist_ok=True)
        paths[subdir] = dir_path
        
    return paths

def save_keypoint_visualization(img, keypoints, filepath, index):
    """
    Save keypoint visualization.
    
    :param img: Input image
    :param keypoints: Detected keypoints
    :param filepath: Path to save the visualization
    :param index: Image index for filename
    """
    img_with_keypoints = cv2.drawKeypoints(
        img, 
        keypoints, 
        None, 
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    cv2.imwrite(os.path.join(filepath, f'keypoints_img_{index}.jpg'), img_with_keypoints)

def save_matches_visualization(left_img, right_img, good_matches, paths, pair_index):
    """
    Save visualization of matches between image pairs.
    
    :param left_img: Left image
    :param right_img: Right image
    :param good_matches: List of good matches
    :param paths: Directory paths
    :param pair_index: Index of the image pair
    """
    h1, w1 = left_img.shape[:2]
    h2, w2 = right_img.shape[:2]
    vis_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    vis_img[:h1, :w1] = left_img
    vis_img[:h2, w1:w1+w2] = right_img
    
    # Draw matches
    for match in good_matches:
        x1, y1 = int(match[0]), int(match[1])
        x2, y2 = int(match[2] + w1), int(match[3])
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.line(vis_img, (x1, y1), (x2, y2), color, 1)
        cv2.circle(vis_img, (x1, y1), 3, color, -1)
        cv2.circle(vis_img, (x2, y2), 3, color, -1)
    
    cv2.imwrite(os.path.join(paths['matches'], f'matches_pair_{pair_index}.jpg'), vis_img)

def get_keypoints(left_img, right_img, paths, pair_index):
    """
    Extract keypoints and save visualizations.
    """
    
    # Convert images to grayscale
    l_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    r_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # Detect and compute keypoints
    sift = cv2.SIFT_create()

    # Compute keypoints
    key_points1, descriptor1 = sift.detectAndCompute(l_img, None)
    key_points2, descriptor2 = sift.detectAndCompute(r_img, None)
    
    # Save keypoint visualizations
    save_keypoint_visualization(left_img, key_points1, paths['keypoints'], f"{pair_index}_left")
    save_keypoint_visualization(right_img, key_points2, paths['keypoints'], f"{pair_index}_right")

    return key_points1, descriptor1, key_points2, descriptor2

def match_keypoints(key_points1, key_points2, descriptor1, descriptor2):
    """
    Match descriptors between images.
    """
    
    # Match descriptors
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    # Apply ratio test we have chosen k = 2 because we want to get the best 2 matches
    matches = bf.knnMatch(descriptor1, descriptor2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            left_pt = key_points1[m.queryIdx].pt
            right_pt = key_points2[m.trainIdx].pt
            good_matches.append([left_pt[0], left_pt[1], right_pt[0], right_pt[1]])
    
    return good_matches

def ransac(good_matches):
    """
    Estimate homography using RANSAC.
    """
    best_inliers = []
    final_H = None
    t = 5

    for _ in range(5000):
        random_pts = random.sample(good_matches, 4)
        H = homography(random_pts)
        inliers = []

        for pt in good_matches:
            p = np.array([pt[0], pt[1], 1]).reshape(3, 1)
            p_1 = np.array([pt[2], pt[3], 1]).reshape(3, 1)
            Hp = np.dot(H, p)
            Hp /= Hp[2]

            if np.linalg.norm(p_1 - Hp) < t:
                inliers.append(pt)

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            final_H = H
    
    return final_H

def stitch_images(image_list):
    """
    Stitches multiple images into a single panorama and saves visualizations.
    
    :param image_list: List of images to be stitched in order
    :return: Stitched panorama image
    """
    if len(image_list) < 2:
        raise ValueError("At least two images are required for stitching.")
    
    # Create directories for this stitching session
    paths = create_session_directories()
    
    stitched_img = image_list[0]
    for i in range(1, len(image_list)):
        # Save original images
        cv2.imwrite(os.path.join(paths['base'], f'original_left_{i}.jpg'), stitched_img)
        cv2.imwrite(os.path.join(paths['base'], f'original_right_{i}.jpg'), image_list[i])
        
        # Process images and save visualizations
        key_points1, descriptor1, key_points2, descriptor2 = get_keypoints(
            stitched_img, image_list[i], paths, i
        )
        
        good_matches = match_keypoints(
            key_points1, key_points2, descriptor1, descriptor2
        )
        
        # Save matches visualization
        save_matches_visualization(stitched_img, image_list[i], good_matches, paths, i)
        
        # Compute homography with RANSAC
        final_H = ransac(good_matches)
        
        # Stitch images
        stitched_img = solution(stitched_img, image_list[i], final_H, paths, i)
    
    final_path = os.path.join(paths['final'], 'final_panorama.jpg')
    cv2.imwrite(final_path, stitched_img)
    
    return stitched_img

def create_blend_mask(img_shape):
    """
    Create a gradient blend mask for smooth transitions.
    
    :param img_shape: Shape of the image (height, width)
    :return: Gradient mask
    """
    height, width = img_shape[:2]
    mask = np.zeros((height, width), dtype=np.float32)
    
    # Create horizontal gradient
    for i in range(width):
        mask[:, i] = i / width
        
    return mask

def solution(left_img, right_img, final_H, paths, pair_index):
    """
    Stitch two images using the computed homography with intensity blending.
    """
    rows1, cols1 = right_img.shape[:2]
    rows2, cols2 = left_img.shape[:2]

    # Calculate the bounds of the stitched image
    points1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    points2 = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)
    
    points2_transformed = cv2.perspectiveTransform(points2, final_H)
    list_of_points = np.concatenate((points1, points2_transformed), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]]) @ final_H

    # Warp the left image
    output_img = cv2.warpPerspective(left_img, H_translation, (x_max - x_min, y_max - y_min))
    
    # Create a mask for the overlapping region
    left_mask = np.zeros((y_max - y_min, x_max - x_min), dtype=np.float32)
    left_mask = cv2.warpPerspective(np.ones(left_img.shape[:2], dtype=np.float32),
                                   H_translation,
                                   (x_max - x_min, y_max - y_min))
    
    # Create right image mask
    right_mask = np.zeros((y_max - y_min, x_max - x_min), dtype=np.float32)
    right_mask[-y_min:rows1 - y_min, -x_min:cols1 - x_min] = 1
    
    # Create blend mask for the overlapping region
    overlap_mask = cv2.multiply(left_mask, right_mask)
    blend_mask = create_blend_mask((y_max - y_min, x_max - x_min))
    
    # Extend right image to match output dimensions
    right_img_extended = np.zeros_like(output_img)
    right_img_extended[-y_min:rows1 - y_min, -x_min:cols1 - x_min] = right_img
    
    # Convert images to float32 for blending
    output_img_float = output_img.astype(np.float32)
    right_img_extended_float = right_img_extended.astype(np.float32)
    
    # Blend the images
    for c in range(3):  #
         
        # Prepare masks for this channel
        inverse_blend = 1 - blend_mask
        
        # Calculate blended parts
        left_part = output_img_float[:, :, c] * inverse_blend * overlap_mask
        right_part = right_img_extended_float[:, :, c] * blend_mask * overlap_mask
        
        # Combine non-overlapping and overlapping regions
        output_img[:, :, c] = (
            output_img_float[:, :, c] * (1 - overlap_mask) + 
            right_img_extended_float[:, :, c] * (right_mask - overlap_mask) +  
            left_part + right_part  
        ).astype(np.uint8)
    
    cv2.imwrite(os.path.join(paths['transformed'], f'stitched_pair_{pair_index}.jpg'), output_img)
    
    return output_img

def homography(points):
    """
    Compute homography matrix using SVD.
    """
    A = []
    for pt in points:
        x, y = pt[0], pt[1]
        X, Y = pt[2], pt[3]
        A.append([x, y, 1, 0, 0, 0, -X * x, -X * y, -X])
        A.append([0, 0, 0, x, y, 1, -Y * x, -Y * y, -Y])

    A = np.array(A)
    _, _, vh = np.linalg.svd(A)
    H = vh[-1, :].reshape(3, 3)
    return H / H[2, 2]

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    
    image_filenames = ['Test/Room/1.jpeg', 'Test/Room/3.jpeg']
    images = [cv2.imread(img) for img in image_filenames]

    if any(img is None for img in images):
        raise FileNotFoundError("One or more image files not found. Check paths.")

    result_img = stitch_images(images)