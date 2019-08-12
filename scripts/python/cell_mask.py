import numpy as np
import zarr
from skimage.filters.rank import entropy
from scipy.ndimage.filters import gaussian_filter
from dask.array import from_array
from functools import partial
from skimage.transform import rescale

data_dir = "/groups/hess/hess_collaborators/Annotations/ParentFiles_whole-cell_images/Jurkat_Cell1_4x4x4nm/Jurkat_Cell1_FS96-Area1_4x4x4nm.n5"

data = from_array(zarr.open(zarr.N5Store(data_dir), mode="r")["volumes/raw/"])

downsampling = 2
entropy_filter_size = 20
gaussian_blur_size = 5


def downsample(img, downsamping):
    from skimage.transform import rescale

    scale = img.ndim * (1 / downsampling,)
    return rescale(img, scale=scale, preserve_range=True, multichannel=False).astype(
        img.dtype
    )


dsfilt = partial(rescale, scale=(), preserve_range=True, multichannel=False)
entfilt = partial(entropy, selem=np.ones(2 * [entropy_filter_size]))
gaufilt = partial(gaussian_filter, sigma=gaussian_blur_size)

if __name__ == "__main__":
    sample = data[0].compute()
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    plt.imshow(sample)
    plt.show()
