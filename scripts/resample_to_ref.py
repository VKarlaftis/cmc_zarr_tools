#!/usr/bin/env python

### Resample 2D slices to reference grid (retain high resolution)
import sys

def main():
    nargs = len(sys.argv)-1
    if nargs<3:
        print("\nUsage: resample_to_ref <input_slice> <ref_slice> <output_slice>\n")
        exit(1)

    # Do the imports after the help
    from pathlib import Path
    from fsl.data.image import Image
    from scipy.ndimage import map_coordinates
    import numpy as np

    slice_file = Path(sys.argv[1])
    ref_file = Path(sys.argv[2])
    output_file = Path(sys.argv[3])

    # Load reference once
    ref_img_nib = Image(str(ref_file))
    ref_affine = ref_img_nib.getAffine('voxel', 'world')
    ref_shape = ref_img_nib.shape
    if len(ref_shape) != 3:
        raise ValueError(f'Reference must be 3D, got shape={ref_shape}')

    # Build reference 2D grid once (slice at x=0), then convert to world coords once
    ref_2d_shape = (ref_shape[1], ref_shape[2])
    y_ref, z_ref = np.meshgrid(
        np.arange(ref_2d_shape[0], dtype=np.float32),
        np.arange(ref_2d_shape[1], dtype=np.float32),
        indexing='ij'
    )
    x_ref = np.zeros_like(y_ref)
    coords_ref_flat = np.array([x_ref, y_ref, z_ref, np.ones_like(x_ref)], dtype=np.float32).reshape(4, -1)
    coords_world = ref_affine @ coords_ref_flat

    slice_img_nib = Image(str(slice_file))
    slice_data = np.asarray(slice_img_nib.data, dtype=np.float32).squeeze()

    # Convert world coords to this slice voxel space
    slice_affine_inv = slice_img_nib.getAffine("world", "voxel")
    coords_slice_flat = slice_affine_inv @ coords_world
    coords_slice_2d = coords_slice_flat[1:3, :].reshape(2, *ref_2d_shape)

    output_array_2d = map_coordinates(
                        slice_data,
                        coords_slice_2d,
                        order=3,
                        mode='constant',
                        cval=0.0,
                      ).astype(np.float32, copy=False)

    # Expand back to 3D for output (with x=0)
    output_array_3d = np.zeros(ref_shape, dtype=np.float32)
    output_array_3d[0, :, :] = output_array_2d

    output_img = Image(output_array_3d, header=ref_img_nib.header)
    output_img.save(str(output_file))


if __name__ == "__main__":
    main()
