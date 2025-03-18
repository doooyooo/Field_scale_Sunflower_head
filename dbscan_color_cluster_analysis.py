import os

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
os.environ["OMP_NUM_THREADS"] = '1'
import ast
import laspy
import pyproj
from osgeo import gdal
from scipy.spatial import distance, Delaunay
from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = 'Times New Roman'

def calculate_bbox_properties(xmin, ymin, xmax, ymax, proj):
    # Convert coordinates to meters
    xmin_m, ymin_m = proj(xmin, ymin)
    xmax_m, ymax_m = proj(xmax, ymax)

    # Convert from meters to centimeters
    xmin_cm, ymin_cm = xmin_m * 100, ymin_m * 100
    xmax_cm, ymax_cm = xmax_m * 100, ymax_m * 100

    # Calculate width and height (in centimeters)
    width_cm = xmax_cm - xmin_cm
    height_cm = ymax_cm - ymin_cm

    # Calculate area (in square centimeters)
    area_cm2 = width_cm * height_cm

    # Calculate diagonal length (in centimeters)
    diagonal_length_cm = np.sqrt(width_cm ** 2 + height_cm ** 2)

    # Return area and diagonal length
    return area_cm2, diagonal_length_cm

def compute_volume_with_delaunay(points,rgb):
    """
    Calculate the volume of point cloud using Delaunay triangulation and visualize tetrahedrons.

    Parameters:
    - points: A NumPy array of shape (N, 3) representing N 3D point coordinates.

    Returns:
    - volume: Calculated enclosed volume of the point cloud.
    """
    # Create Delaunay triangulation
    delaunay = Delaunay(points)

    # Calculate volume of Delaunay tetrahedrons
    tetrahedra = points[delaunay.simplices]

    # Vectorized calculation of tetrahedron volumes
    v0 = tetrahedra[:, 0, :]
    v1 = tetrahedra[:, 1, :]
    v2 = tetrahedra[:, 2, :]
    v3 = tetrahedra[:, 3, :]

    # Calculate tetrahedron volumes
    volume = np.abs(np.einsum('ij,ij->i',
                              np.cross(v1 - v0, v2 - v0),
                              v3 - v0)) / 6.0
    total_volume = np.sum(volume)
    return total_volume

for file_name in file_names_without_extension:
    print(f"Processing {file_name}")
    # Also need to open TIF file to obtain coordinates
    # Open TXT file and read annotation boxes
    # Open point cloud file
    txt_f = os.path.join(txt_path, file_name + '.txt')
    point_f = os.path.join(point_path, file_name + '.las')
    tif_f = os.path.join(tif_path, file_name + '.tif')

    # Read .txt file
    with open(txt_f, 'r') as f:
        txt_data = f.readlines()
        print(f"Total of {len(txt_data)} detection boxes")  # Read all lines

    with (laspy.open(point_f) as las_file):
        las_data = las_file.read()
        # Extract X, Y, Z values
        las_x = las_data.x
        las_y = las_data.y
        las_z = las_data.z
        las_r = np.array(las_data.red / 65535 * 255)  # Reflectance needs to be mapped to 0-255
        las_g = np.array(las_data.green / 65535 * 255)
        las_b = np.array(las_data.blue / 65535 * 255)

    # Read .tif file using GDAL
    dataset = gdal.Open(tif_f)  # Open TIFF file
    if dataset is None:
        print("Failed to open file: ", tif_f)
    else:
        # Read image data
        tif_data = dataset.ReadAsArray()  # Read data as NumPy array

        # Get data dimensions and metadata
        geo_transform = dataset.GetGeoTransform()  # Geographic transformation info
        top_left_x = geo_transform[0]
        pixel_width = geo_transform[1]
        top_left_y = geo_transform[3]
        pixel_height = geo_transform[5]

    # Process each detection box in txt file for clustering and parameter calculation
    areas, diagonals, volumns, distances_ = [],[],[],[]

    i = 0
    for txt_line in txt_data:
        txt_line = txt_line.strip()
        type = int(txt_line[0])
        jiancekuang = ast.literal_eval(txt_line[1:])

        if type==1:
            continue
        # Split into (X, Y) pairs
        x_coords = jiancekuang[0::2]  # Get X coordinates (even indices)
        y_coords = jiancekuang[1::2]  # Get Y coordinates (odd indices)

        # Convert XY coordinates to geographic coordinates using GeoTransform
        geographic_coords = []
        for x, y in zip(x_coords, y_coords):
            # Convert each XY pair to geographic coordinates
            geo_x = top_left_x + (x * pixel_width)  # Longitude
            geo_y = top_left_y + (y * pixel_height)  # Latitude

            geographic_coords.append((geo_x, geo_y))

        xmin, xmax = geographic_coords[0][0],geographic_coords[1][0]
        ymax, ymin = geographic_coords[0][1],geographic_coords[1][1]
        bbox_min = np.array([xmin, ymin])  # Replace with actual min coordinates
        bbox_max = np.array([xmax, ymax])  # Replace with actual max coordinates

        # Find points within the bounding box
        in_bbox = ((las_x >= bbox_min[0]) & (las_x <= bbox_max[0]) &
                   (las_y >= bbox_min[1]) & (las_y <= bbox_max[1]))

        # Convert latitude/longitude to meters using pyproj
        proj = pyproj.Proj(proj='utm', zone=49, ellps='WGS84')

        x_meters, y_meters = proj(las_x[in_bbox], las_y[in_bbox])
        # Convert to centimeters
        x_cm = x_meters * 100
        y_cm = y_meters * 100
        z_cm = las_z[in_bbox] * 100
        X = np.column_stack((x_cm, y_cm, z_cm))
        RGB = np.column_stack((las_r[in_bbox], las_g[in_bbox], las_b[in_bbox]))

        # Design features
        # Filter features before clustering to remove noise
        feature = []
        if type==0:
            # An index similar to yellow flower index, normalizing green and blue bands, high values are targets
            feature = (RGB[:,1]-RGB[:,2])/(RGB[:,1]+RGB[:,2])*(RGB[:,1]/max(RGB[:,1]))
        elif type==2:
            # An index similar to yellow flower index, normalizing green and red bands, low values are targets
            feature = ((RGB[:,1]-RGB[:,0])/(RGB[:,1]+RGB[:,0]))*(max(RGB[:,0])/RGB[:,0])

        # Reshape for KMeans (1D to 2D)
        feature_reshaped = feature.reshape(-1, 1)
        # Initialize KMeans model
        kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto')
        # Apply feature to KMeans model
        if type==0:
            kmeans.fit(feature_reshaped)
            labels = kmeans.labels_
        else:
            kmeans.fit(feature_reshaped[target_indices])  # Cluster only on qualified indices
            labels = np.full(feature.shape, -1)  # Default all labels to non-target class
            labels[target_indices] = kmeans.labels_  # Assign only target class labels

        # Calculate cluster means
        cluster_0_mean = np.mean(feature[labels == 0])
        cluster_1_mean = np.mean(feature[labels == 1])
        # For flowers, high values are the target class; for maturity, low values are targets
        if type == 0:
            if cluster_1_mean > cluster_0_mean:
                targets_type = 1
            else:
                targets_type = 0
        elif type == 2:
            if abs(cluster_1_mean) < abs(cluster_0_mean):
                targets_type = 1
            else:
                targets_type = 0
        if targets_type==1:
            non_targets = 0
        else:
            non_targets = 1

        # Filter points belonging to targets_type
        target_indices = np.where(labels == targets_type)[0]
        target_features = feature[labels == targets_type]
        first_label = labels.copy()
        # Extract corresponding Z values
        z_values = X[target_indices, 2].reshape(-1, 1)
        # Perform clustering
        kmeans = DBSCAN(eps=0.5, min_samples=10)
        kmeans.fit(z_values)
        labels_ = kmeans.labels_

        # Calculate volume using voxelization method
        voxel_size = 0.8  # Voxel size in cm
        volume = compute_volume_with_delaunay(target_points,target_points_RGB)
        volumns.append(volume)
        print(f"Point cloud volume: {volume}")

        # Calculate maximum distance between points
        if target_points.size == 0:
            print("No target points found. Cannot compute maximum distance.")
        else:
            max_distance = distance.pdist(target_points, 'euclidean').max()
            distances_.append(max_distance)
