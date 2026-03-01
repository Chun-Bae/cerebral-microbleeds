from src.core.data_procssing import (
    run_skull_stripping,
    run_n4_correction,
    run_nii_to_png,
    run_extract_bboxes,
    run_generate_splits_fixed,
    run_generate_splits_kfold,
    run_generate_lmdb_fixed,
    run_generate_lmdb_kfold,
)


class DataPipeLine:
    def __init__(self):
        pass

    def run(self):
        self.process_skull_stripping()
        self.process_n4_correction()
        self.process_nii_to_png()
        self.extract_bboxes()
        self.generate_splits_patients()
        self.generate_lmdb_dataset()

    def process_skull_stripping(self):
        run_skull_stripping()

    def process_n4_correction(self):
        run_n4_correction()

    def process_nii_to_png(self):
        run_nii_to_png()

    def extract_bboxes(self):
        run_extract_bboxes()

    def generate_splits_patients(self):
        run_generate_splits_fixed()
        run_generate_splits_kfold()

    def generate_lmdb_dataset(self):
        run_generate_lmdb_fixed()
        run_generate_lmdb_kfold()
