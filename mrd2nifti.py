#!/usr/bin/python3

"""
Convert an MRD image .h5 file into a NIfTI (.nii.gz) file, for viewing in standard
neuroimaging tools (ITK-SNAP, FSLeyes, MRIcroGL, etc.). Unlike mrd2gif.py (a quick 2D
mosaic preview), this preserves real-world voxel spacing and orientation via a proper
NIfTI affine, built from the MRD ImageHeader's position/read_dir/phase_dir/slice_dir.
"""

import argparse
import h5py
import ismrmrd
import numpy as np
import nibabel as nib
import mrdhelper

defaults = {
    'in_group': '',
    'series':   '',
}


def build_affine(img, voxel_mm):
    """
    Build a NIfTI affine from one MRD image's geometry. MRD/DICOM orientation vectors
    are in the LPS (Left-Posterior-Superior) patient coordinate system; NIfTI expects
    RAS (Right-Anterior-Superior), so the x and y components are negated when copied in.
    """
    read_dir  = np.array(img.read_dir)
    phase_dir = np.array(img.phase_dir)
    slice_dir = np.array(img.slice_dir)
    position  = np.array(img.position)

    affine = np.eye(4)
    # Columns: image axes (row=phase_dir, col=read_dir, slice=slice_dir) scaled by voxel size
    affine[:3, 0] = phase_dir * voxel_mm[0] * [-1, -1, 1]
    affine[:3, 1] = read_dir  * voxel_mm[1] * [-1, -1, 1]
    affine[:3, 2] = slice_dir * voxel_mm[2] * [-1, -1, 1]
    affine[:3, 3] = position  * [-1, -1, 1]
    return affine


def main(args):
    with h5py.File(args.filename, 'r') as d:
        dsetNames = list(d.keys())
        if not args.in_group:
            if len(dsetNames) == 1:
                args.in_group = dsetNames[0]
            else:
                raise Exception(f"Input group not specified and multiple groups are present: {dsetNames}")

    print(f"Reading data from group '{args.in_group}' in file '{args.filename}'")

    with ismrmrd.Dataset(args.filename, args.in_group, False) as dset:
        subgroups = dset.list()
        imgGroups = [g for g in subgroups if g.startswith('image_') or g.startswith('images_')]

        if args.series:
            imgGroups = [g for g in imgGroups if g == args.series]
        if not imgGroups:
            raise Exception(f"No matching image series found. Available: {[g for g in subgroups if g.startswith('image_')]}")

        for group in imgGroups:
            n = dset.number_of_images(group)
            images = [dset.read_image(group, i) for i in range(n)]

            # Sort by slice index so the NIfTI stacks in physical order
            images.sort(key=lambda im: im.slice)

            data = np.stack([np.squeeze(im.data) for im in images], axis=-1)  # (rows, cols, nSlices)

            meta0 = ismrmrd.Meta.deserialize(images[0].attribute_string)
            pixelSpacing   = mrdhelper.get_meta_value(meta0, 'PixelSpacing')
            sliceThickness = mrdhelper.get_meta_value(meta0, 'SliceThickness')
            if pixelSpacing is not None and sliceThickness is not None:
                voxel_mm = [float(pixelSpacing[0]), float(pixelSpacing[1]), float(sliceThickness)]
            else:
                # Fallback: MRD field_of_view is total FOV in mm; matrix size is 1 slice thick
                # per image, so field_of_view[2] is already the slice thickness directly.
                fov = images[0].field_of_view
                voxel_mm = [float(fov[1]) / data.shape[0], float(fov[0]) / data.shape[1], float(fov[2])]

            affine = build_affine(images[0], voxel_mm)

            out_path = f"{args.filename.rsplit('.', 1)[0]}_{args.in_group}_{group}.nii.gz"
            nib.save(nib.Nifti1Image(data.astype(np.float32), affine), out_path)
            print(f"  Wrote {group} ({n} slices, voxel size {voxel_mm} mm) -> {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert an MRD image file to NIfTI',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filename',                help='Input MRD .h5 file')
    parser.add_argument('-g', '--in-group',        help='Input data group (default: auto-detect if only one present)')
    parser.add_argument('-s', '--series',          help='Specific image series to convert (default: all series)')
    parser.set_defaults(**defaults)
    args = parser.parse_args()
    main(args)
