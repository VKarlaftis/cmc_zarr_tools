#!/usr/bin/env python

# Create a 3D stack from 2D resampled images
import sys


def main():
    nargs = len(sys.argv)-1
    if nargs<3:
        print("\nUsage: create_3d_zarr <input_dir> <ref_slice> <output_zarr>\n")
        exit(1)
    
    # Do the imports after the help
    from pathlib import Path
    import zarr
    import numpy as np
    import nibabel as nib

    slice_dir = Path(sys.argv[1])
    ref_file = Path(sys.argv[2])
    output_file = Path(sys.argv[3])

    slice_files = sorted(slice_dir.glob('*.nii.gz'))

    if len(slice_files) == 0:
        raise ValueError(f'No slices found in {slice_dir}')

    # Load reference metadata once
    ref_img_nib = nib.load(str(ref_file))
    ref_affine = ref_img_nib.affine.copy()

    # Create output Zarr group
    root = zarr.open_group(str(output_file), mode='w', zarr_format=2)

    # Input slices are stored as 3D volumes with data in the first x-plane
    first_img = nib.load(str(slice_files[0]))
    first_plane = np.asarray(first_img.dataobj[0, :, :], dtype=np.float32)
    y, z = first_plane.shape
    x = len(slice_files)

    # Store as (Z, Y, X) to match axis annotation [z, y, x]
    arr0 = root.create_array(
        '0',
        shape=(z, y, x),
        chunks=(min(1024, z), min(1024, y), 1),
        dtype='float32',
    )

    # Extract voxel size and origin
    voxel_x = float(np.linalg.norm(ref_affine[:3, 0]))
    voxel_y = float(np.linalg.norm(ref_affine[:3, 1]))
    voxel_z = float(np.linalg.norm(ref_affine[:3, 2]))
    origin = ref_affine[:3, 3].copy()

    # Build diagonal affine
    affine_xyz = np.array([
        [voxel_x, 0.0, 0.0, float(origin[0])],
        [0.0, voxel_y, 0.0, float(origin[1])],
        [0.0, 0.0, voxel_z, float(origin[2])],
    ], dtype=np.float64)

    # Orientation metadata
    orientation = {'x': 'l', 'y': 'a', 'z': 's'}

    # Initialise multiscales
    max_levels = 6
    xy_switch_threshold = 512
    min_yz_size = 32

    levels = []
    cum_fx, cum_fy, cum_fz = 1, 1, 1
    cur_x, cur_y, cur_z = x, y, z

    for level_idx in range(1, max_levels + 1):
        if min(cur_y, cur_z) < min_yz_size:
            break

        use_x_downsample = (min(cur_y, cur_z) <= xy_switch_threshold) and (cur_x >= 2)
        step_fx = 2 if use_x_downsample else 1
        step_fy = 2 if cur_y >= 2 else 1
        step_fz = 2 if cur_z >= 2 else 1

        if (step_fx, step_fy, step_fz) == (1, 1, 1):
            break

        cum_fx *= step_fx
        cum_fy *= step_fy
        cum_fz *= step_fz

        out_x = x // cum_fx
        out_y = y // cum_fy
        out_z = z // cum_fz
        if out_x == 0 or out_y == 0 or out_z == 0:
            break

        level_name = str(level_idx)
        level_array = root.create_array(
            level_name,
            shape=(out_z, out_y, out_x),
            chunks=(min(512, out_z), min(512, out_y), 1),
            dtype='float32',
        )

        levels.append({
            'name': level_name,
            'array': level_array,
            'cum_fx': cum_fx,
            'cum_fy': cum_fy,
            'cum_fz': cum_fz,
            'out_x': out_x,
            'out_y': out_y,
            'out_z': out_z,
            'x_trim': out_x * cum_fx,
            'y_trim': out_y * cum_fy,
            'z_trim': out_z * cum_fz,
            'buffer': np.zeros((out_z, out_y), dtype=np.float32),
            'buffer_count': 0,
            'x_out_idx': 0,
        })

        cur_x, cur_y, cur_z = out_x, out_y, out_z

    # ==== Main loop: load slices and update zarr ====
    overall_min = float('inf')
    overall_max = float('-inf')

    for i, p in enumerate(slice_files):
        img = nib.load(str(p))
        data_yz = np.asarray(img.dataobj[0, :, :], dtype=np.float32)

        if data_yz.shape != (y, z):
            raise ValueError(f'shape mismatch for {p.name}: {data_yz.shape} != {(y, z)}')

        # Store directly to Zarr: data is (Y, Z), store transposed to arr0[Z, Y, X]
        arr0[:, :, i] = data_yz.T
        
        # Track min/max
        smin = float(data_yz.min())
        smax = float(data_yz.max())
        if smin < overall_min:
            overall_min = smin
        if smax > overall_max:
            overall_max = smax

        # Write pyramid levels from the same slice
        for lvl in levels:
            if i >= lvl['x_trim']:
                continue

            d2 = data_yz[:lvl['y_trim'], :lvl['z_trim']]
            yz_down = d2.reshape(
                lvl['out_y'], lvl['cum_fy'], lvl['out_z'], lvl['cum_fz']
            ).mean(axis=(1, 3), dtype=np.float32)

            if lvl['cum_fx'] == 1:
                lvl['array'][:, :, i] = yz_down.T
            else:
                lvl['buffer'] += yz_down.T
                lvl['buffer_count'] += 1
                if lvl['buffer_count'] == lvl['cum_fx']:
                    lvl['array'][:, :, lvl['x_out_idx']] = (lvl['buffer'] / lvl['cum_fx']).astype(np.float32, copy=False)
                    lvl['x_out_idx'] += 1
                    lvl['buffer'].fill(0)
                    lvl['buffer_count'] = 0

    # Metadata
    root.attrs['data_min'] = float(overall_min)
    root.attrs['data_max'] = float(overall_max)

    datasets = []
    for idx in range(0, len(levels) + 1):
        if idx == 0:
            fx = fy = fz = 1
            path = '0'
        else:
            lvl = levels[idx - 1]
            fx, fy, fz = lvl['cum_fx'], lvl['cum_fy'], lvl['cum_fz']
            path = lvl['name']

        sx = voxel_x * fx
        sy = voxel_y * fy
        sz = voxel_z * fz

        tx = 0.5 * (fx - 1) * voxel_x
        ty = 0.5 * (fy - 1) * voxel_y
        tz = 0.5 * (fz - 1) * voxel_z

        datasets.append({
            'path': path,
            'coordinateTransformations': [
                {'type': 'scale', 'scale': [sz, sy, sx]},
                {'type': 'translation', 'translation': [tz, ty, tx]},
            ],
        })

    root.attrs['multiscales'] = [{
        'axes': [
            {'name': 'z', 'type': 'space', 'unit': 'millimeter'},
            {'name': 'y', 'type': 'space', 'unit': 'millimeter'},
            {'name': 'x', 'type': 'space', 'unit': 'millimeter'},
        ],
        'coordinateTransformations': [
            {'type': 'scale', 'scale': [1.0, 1.0, 1.0]}
        ],
        'datasets': datasets,
        'name': '',
        'type': 'median window 2x2x2',
        'version': '0.4',
    }]

    # Store nifti metadata
    nifti_group = root.create_group('nifti')
    nifti_group.create_array('0', shape=(348,), chunks=(348,), dtype='uint8')

    nifti_group.attrs.update({
        'Affine': affine_xyz[:3, :4].tolist(),
        'Dim': [int(x), int(y), int(z)],
        'VoxelSize': [voxel_x, voxel_y, voxel_z],
        'Orientation': orientation,
        'SForm': 'aligned_anat',
        'QForm': '',
        'QuaternOffset': {
            'x': float(origin[0]),
            'y': float(origin[1]),
            'z': float(origin[2]),
        },
        'Unit': {'L': 'mm', 'T': 's'},
        'DataType': 'single',
        'BitDepth': 32,
        'NIIHeaderSize': 348,
    })


if __name__ == "__main__":
    main()
