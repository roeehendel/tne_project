import gzip
import os.path
import shutil

from data_loading.tne_data_paths import GZIP_EXTENSION, TRAIN_DATASET_COMPRESSED, DEV_DATASET_COMPRESSED, \
    TEST_DATASET_COMPRESSED


def decompress_file(in_file_path, out_file_path):
    with gzip.open(in_file_path, 'rb') as f_in:
        with open(out_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def decompress_tne_dataset():
    for in_file_path in [TRAIN_DATASET_COMPRESSED, DEV_DATASET_COMPRESSED, TEST_DATASET_COMPRESSED]:
        out_file_path = in_file_path.replace(GZIP_EXTENSION, '')

        if not os.path.isfile(out_file_path):
            decompress_file(in_file_path, out_file_path)


if __name__ == '__main__':
    decompress_tne_dataset()
