import sys
import os
import nibabel as nib


def check_nii(nii_path):
    if not os.path.exists(nii_path):
        print(f"Error: File not found at {nii_path}")
        return

    nii_img = nib.load(nii_path)
    print(f"--- NIfTI Header Info: {os.path.basename(nii_path)} ---")
    print(nii_img.header)
    print("\n--- Data Shape ---")
    print(nii_img.shape)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_nii.py <path_to_nii_file>")
    else:
        check_nii(sys.argv[1])