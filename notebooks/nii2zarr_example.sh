#!/bin/bash

psoct='/Users/Vasilis/Downloads/CMC_results/Moe_cc_Ret/slide_deck_to_mri.nii.gz' #Retardance/lowres/Slice_111_EnR_downsample_10_hdr_float32.nii.gz'
dti='/Users/Vasilis/CMC_data/Moe/MRI/reoriented_FA.nii.gz'

ngff-zarr -i ${dti} -o Moe_cc/DTI_FA.zarr --ome-zarr-version 0.5
ngff-zarr -i ${psoct} -o Moe_cc/PSOCT_Ret_in_DTI.zarr --ome-zarr-version 0.5

python visualize_zarr_ng/visualize_zarr.py Moe_cc/DTI_FA.zarr Moe_cc/PSOCT_Ret_in_DTI.zarr --port 8080 --name DTI_native PSOCT_Ret_in_DTI
