from typing import Union, Optional
from matplotlib.colors import Colormap
import numpy as np
from numpy.typing import NDArray

def parse_color(
    color: Union[tuple, list, str, float, int],
    cmap: Optional[Union[str, Colormap]] = 'viridis'
) -> NDArray[np.float32]:
    """
    Convert a variety of color specifications to an RGBA array in [0, 1].
    
    Parameters:
        color: tuple/list, str, or float
            - RGB tuple/list (0-1 floats or 0-255 ints)
            - Named color (string) like 'red', 'skyblue'
            - Float in [0,1]: maps through the given colormap
        cmap: str or Colormap
            - Colormap name or Colormap object, used if color is a float
            
    Returns:
        np.ndarray of shape (4,), RGBA in [0,1]
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors

    if cmap is None:
        cmap = 'viridis'

    # If float, use colormap
    if isinstance(color, (int, float)):
        if not (0 <= color <= 1):
            raise ValueError(f"Float color must be between 0 and 1, got {color}")
        cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
        return np.array(cmap_obj(color))

    # If tuple/list
    if isinstance(color, (tuple, list)) and len(color) == 3:
        arr = np.array(color, dtype=float)
        if arr.max() > 1:  # Assume it's in 0-255
            arr = arr / 255.0
        return np.append(arr, 1.0)  # Add alpha=1

    # If string (named color, hex, etc.)
    if isinstance(color, str):
        return np.array(colors.to_rgba(color))

    raise TypeError(f"Unsupported color type: {type(color)}")
