import cv2
import numpy as np
from tqdm import tqdm
import open3d as o3d
from sklearn.decomposition import PCA
import glob
import os

def from_pcd(pcd_filename, shape, default_filler=0, index=None):
        img = np.zeros(shape)
        if default_filler != 0:
            img += default_filler

        with open(pcd_filename) as f:
            for l in f.readlines():
                ls = l.split()

                if len(ls) != 5:
                    continue
                try:
                    float(ls[0])
                except ValueError:
                    continue

                i = int(ls[4])
                r = i // shape[1]
                c = i % shape[1]

                if index is None:
                    x = float(ls[0])
                    y = float(ls[1])
                    z = float(ls[2])

                    img[r, c] = np.sqrt(x ** 2 + y ** 2 + z ** 2)

                else:
                    img[r, c] = float(ls[index])

        return img / 1000.0



def task_1_overlay_rects(image_path, pos_rects, neg_rects, save_path=None):
    img = cv2.imread(image_path)
    # rects are usually 4 points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    for rect in pos_rects:
        cv2.polylines(img, [rect], True, (0, 255, 0), 2) # Green for Positive
    for rect in neg_rects:
        cv2.polylines(img, [rect], True, (0, 0, 255), 2) # Red for Negative
    if save_path:
        cv2.imwrite(save_path, img)
    return img

def task_2_create_rgbd(rgb_img, pcd_file_path):
    """
    Project 3D point cloud (x,y,z) to 2D image plane (u,v).
    Note: You will need the camera intrinsics (fx, fy, cx, cy) 
    usually found in the dataset 'utils'.
    """
    # rgb_img = cv2.imread(rgb_img_path) # Shape: (H, W, 3)
    h, w, _ = rgb_img.shape
    
    # 2. Generate Depth Image using your provided logic
    # Assuming 'DepthImage' is the class containing your from_pcd method
    depth_img = from_pcd(pcd_file_path, shape=(h, w)) # Extract the underlying numpy array
    
    # 3. Stack them into a 4-channel RGB-D image
    # Note: OpenCV loads as BGR, you might want to convert to RGB first
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    rgbd = np.dstack((rgb_img, depth_img))
    
    return rgbd
    
def task_3_extract_features(rgbd_patch, pos_rects):
    # Split RGB and Depth
    rgb = rgbd_patch[:, :, :3].astype(np.uint8)
    depth = rgbd_patch[:, :, 3].astype(np.float32)
    
    # Convert RGB to YUV
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, pos_rects, 255)
    masked_rgb = cv2.bitwise_and(rgb, rgb, mask=mask)
    yuv_patch = cv2.cvtColor(masked_rgb, cv2.COLOR_BGR2YUV)
    masked_depth = np.where(mask == 255, depth, 0.0)
    return [(yuv_patch, masked_depth)]

def task_4_pca_whitening(depth_features):
    # Flatten features: (N_patches, Height * Width)
    whitened_depth = depth_features.flatten()
    whitened_depth -= np.mean(whitened_depth) # Centering
    whitened_depth = whitened_depth.reshape(depth_features.shape) # Reshape for PCA
    return whitened_depth

def task_5_visualize_pcd(pcd_file):
    # Load and visualize using Open3D
    pcd = o3d.io.read_point_cloud(pcd_file)
    o3d.visualization.draw_geometries([pcd])

def save_rgbd_to_pcd(yuv_patch, depth_patch, output_filename):
    # 1. Convert YUV back to RGB for standard visualization
    # If you only want the 'Y' channel (Intensity), use that instead
    rgb_patch = cv2.cvtColor(yuv_patch, cv2.COLOR_YUV2RGB)
    
    # 2. Create Open3D Images
    color_o3d = o3d.geometry.Image(rgb_patch.astype(np.uint8))
    depth_o3d = o3d.geometry.Image(depth_patch.astype(np.float32))
    
    # 3. Create RGBD Image
    # We use scale=1.0 because your from_pcd code already divided by 1000.0
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, depth_scale=1.0, convert_rgb_to_intensity=False
    )
    
    # 4. Project to Point Cloud
    # Using standard intrinsics; for the assignment, the default is usually fine
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
    
    # 5. Save to file
    o3d.io.write_point_cloud(output_filename, pcd)

if __name__ == "__main__":

    path = 'C:\\Users\\manas\\.cache\\kagglehub\\datasets\\oneoneliu\\cornell-grasp'
    if not os.path.exists(path):
        print('Downloading dataset from Kaggle...')
        import kagglehub
        # Download latest version
        path = kagglehub.dataset_download("oneoneliu/cornell-grasp")
        print("Path to dataset files:", path)
        
    image_paths = glob.glob(os.path.join(path, "**/*r.png"), recursive=True)

    for img_path in tqdm(image_paths):
        # print("Processing:", img_path)
        base = img_path[:-5]
        pos_file = base + "cpos.txt"
        neg_file = base + "cneg.txt"
        depth_file = base + ".txt"
        
        if not os.path.exists(pos_file):
            continue
        if not os.path.exists(depth_file):
            continue
        pos_rects = np.loadtxt(pos_file).reshape(-1, 4, 2).astype(np.int32) # Reshape to (N, 4, 2)
        neg_rects = np.loadtxt(neg_file).reshape(-1, 4, 2).astype(np.int32) # Reshape to (N, 4, 2)
        rgb_labelled = task_1_overlay_rects(img_path, pos_rects, neg_rects, save_path="images/" + img_path.split("\\")[-1].split(".")[0] + "_overlay.png")
        rgbd = task_2_create_rgbd(cv2.imread(img_path), depth_file)
        pos_rect_imgs = task_3_extract_features(rgbd, pos_rects)
        for i, (yuv_patch, depth_patch) in enumerate(pos_rect_imgs):
            whitened_depth = task_4_pca_whitening(depth_patch)
            save_rgbd_to_pcd(yuv_patch, whitened_depth, "pcds/" + img_path.split("\\")[-1].split(".")[0] + f"_patch_{i}.ply")
        