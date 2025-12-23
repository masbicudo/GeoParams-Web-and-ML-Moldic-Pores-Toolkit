from k_means_constrained import KMeansConstrained
import numpy as np
import numpy as np
from sklearn.cluster import KMeans


def cluster_pixels_kmeans_constrained_model(mask, n_clusters=8, fraction=10):
    coords = np.column_stack(np.where(mask > 0))

    coords_few = coords[::fraction]  # Use a subset for faster computation

    num_pixels = len(coords_few)
    assert num_pixels >= n_clusters * 5, "Too few pixels per cluster"


    # Compute roughly equal cluster size
    size_per_cluster = len(coords_few) // n_clusters

    # Fit balanced KMeans
    kmeans = KMeansConstrained(
        n_clusters=n_clusters,
        size_min=int(size_per_cluster * 0.8),  # small tolerance
        size_max=int(size_per_cluster * 1.2),
        random_state=42
    )
    kmeans.fit(coords_few)

    return kmeans

def cluster_pixels_kmeans_constrained_regions(mask, cp_model):
    coords2 = np.column_stack(np.where(mask > 0))

    # Predict cluster labels for mask using centroids from mask1
    labels2 = cp_model.predict(coords2, size_min=None, size_max=None)
    
    # Visualize result
    regions = np.zeros_like(mask)
    regions[coords2[:, 0], coords2[:, 1]] = labels2 + 1

    return regions





def cluster_pixels_h_splits_model(mask):
    coords = np.column_stack(np.where(mask > 0))

    # Sort coordinates (e.g. by y then x)
    coords = coords[np.lexsort((coords[:, 1], coords[:, 0]))]

    # Split evenly
    n_clusters = 8
    splits = np.array_split(coords, n_clusters)
    split_points = [
            (split[0]  if it > 0               else [0,0],
             split[-1] if it < len(splits) - 1 else [mask.shape[0] - 1, mask.shape[1] - 1]
            ) for it, split in enumerate(splits)
        ]
    return split_points

def cluster_pixels_h_splits_regions(mask, split_points):
    regions = np.zeros_like(mask)
    for i, (start, end) in enumerate(split_points):
        if start[0] == end[0]:
            regions[start[0], start[1]:end[1]+1] = i + 1
        elif start[0] == end[0] - 1:
            regions[start[0], start[1]:] = i + 1
            regions[end[0], :end[1]+1] = i + 1 
        elif start[0] < end[0]:
            regions[start[0], start[1]:] = i + 1
            regions[start[0]+1:end[0], :] = i + 1
            regions[end[0], :end[1]+1] = i + 1
    return regions*mask






def cluster_pixels_kmeans_model(mask, n_regions=8, random_state=42):
    # Get coordinates of white pixels
    coords = np.column_stack(np.where(mask > 0))

    # Run KMeans on coordinates
    kmeans = KMeans(n_clusters=n_regions, random_state=random_state, n_init=10)
    kmeans = kmeans.fit(coords)

    return kmeans

def cluster_pixels_kmeans_regions(mask, cp_model):
    coords2 = np.column_stack(np.where(mask > 0))

    # Predict cluster labels for mask using centroids from mask1
    labels2 = cp_model.predict(coords2)

    # Create a label map for visualization
    regions2 = np.zeros_like(mask, dtype=np.uint8)
    regions2[coords2[:, 0], coords2[:, 1]] = labels2.astype(np.uint8) + 1

    return regions2
