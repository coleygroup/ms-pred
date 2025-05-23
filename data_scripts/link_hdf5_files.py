import argparse
from pathlib import Path
from tqdm import tqdm
import h5py
import logging

import ms_pred.common as common


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default='results/dag_nist20/split_1_rnd1/preds_train_100',
                        action="store", type=str)
    parser.add_argument("--glob-pattern", default='chunk_*/decoy_tree_preds.hdf5',
                        action="store", type=str)
    parser.add_argument("--target-path", default='results/dag_nist20/split_1_rnd1/decoy_tree_preds.hdf5',
                        action="store", type=str)
    return parser.parse_args()


def link_hdf5_files():
    args = get_args()
    kwargs = args.__dict__

    source_path = Path(kwargs['source_dir'])
    source_paths = list(source_path.glob(kwargs['glob_pattern']))
    if len(source_paths) == 0:
        raise ValueError('No HDF5 file found that matches the pattern')

    target_path = Path(kwargs['target_path'])
    tgt_h5 = common.HDF5Dataset(target_path, 'w')
    for source_path in source_paths:
        src_h5 = common.HDF5Dataset(source_path)

        # update root attributes
        for attr_key, attr_val in src_h5.attrs.items():
            tgt_h5.attrs[attr_key] = attr_val

        # link datasets/groups
        for key in src_h5.get_all_names():
            if key in tgt_h5.h5_obj:
                raise ValueError(f'cannot merge multiple groups. key: {key}')
            tgt_h5[key] = h5py.ExternalLink(source_path, key)
        src_h5.close()
    tgt_h5.close()


if __name__ == '__main__':
    import time

    start_time = time.time()
    link_hdf5_files()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")