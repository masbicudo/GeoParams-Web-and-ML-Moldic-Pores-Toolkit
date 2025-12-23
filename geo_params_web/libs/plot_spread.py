from typing import Tuple
from matplotlib import patches
import numpy as np
from sklearn.decomposition import PCA

def plot_spread(ax, xs, ys, 
                          color: str | Tuple[float, float, float, float] | None = "blue", 
                          draw_axes: bool = False):
    """
    Plots an ellipse representing the spread of 2D data points and optionally
    draws its principal axes.
    """
    # Perform PCA
    pca = PCA(n_components=2)
    points = np.column_stack((xs, ys))  # Combine xs and ys into a 2D array
    mean = np.mean(points, axis=0)      # Mean of the points
    
    # Fit PCA to the points. No need to center manually, as PCA in scikit-learn does it.
    pca.fit(points)
    
    # The singular values (sqrt of the eigenvalues of the covariance matrix) 
    # determine the half-lengths of the ellipse's axes (semi-axes).
    # We multiply by 2 for the full width and height of the ellipse patch.
    # A scaling factor (e.g., 2) is used to capture ~95% of the data for a normal distribution.
    scale_factor = 2 
    width, height = np.sqrt(pca.explained_variance_) * scale_factor
    
    # The angle of the ellipse is determined by the direction of the first principal component
    angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
    
    # Create the ellipse with the correct parameters
    ellipse = patches.Ellipse(mean, width * 2, height * 2, 
                              angle=np.degrees(angle), 
                              edgecolor=color, 
                              facecolor='none',
                              linewidth=2)
    
    # Add the ellipse to the plot
    ax.add_patch(ellipse)

    # If requested, draw the principal axes
    if draw_axes:
        # First principal component (major axis)
        v1 = pca.components_[0] * width
        major_axis_start = mean - v1
        major_axis_end = mean + v1
        ax.plot([major_axis_start[0], major_axis_end[0]], 
                [major_axis_start[1], major_axis_end[1]], 
                color=color, linestyle='--', linewidth=1.5, alpha=0.8)

        # Second principal component (minor axis)
        v2 = pca.components_[1] * height
        minor_axis_start = mean - v2
        minor_axis_end = mean + v2
        ax.plot([minor_axis_start[0], minor_axis_end[0]], 
                [minor_axis_start[1], minor_axis_end[1]], 
                color=color, linestyle='--', linewidth=1.5, alpha=0.8)

