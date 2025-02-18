import os
from . import (
    base_dataset,
    arad1k_dataset, cave_dataset, harvard_dataset,
    wdcmall_dataset, paviac_dataset, chikusei_dataset,
    resisc45_dataset, aid_dataset
)

name2dataset_dict = {
    'ARAD1K': arad1k_dataset,
    'CAVE': cave_dataset,
    'Harvard': harvard_dataset,
    'Pavia Centre': paviac_dataset,
    'Washington DC Mall': wdcmall_dataset,
    'Chikusei': chikusei_dataset,
    'NWPU RESISC45': resisc45_dataset,
    'AID': aid_dataset,
}


def get_dataset(
        root_path: str,
        dataset: str,
        train_or_test: str,
        cache: bool,
        return_keys=None
) -> base_dataset.BaseHyperDataset:

    if return_keys is None:
        return_keys = ['hsi', 'rgb', 'svd']

    o_dataset = base_dataset.BaseHyperDataset(
        root_path=os.path.join(root_path, dataset, train_or_test),
        dataset=dataset,
        return_keys=return_keys,
        cache=cache,
        hsi_loader=name2dataset_dict[dataset].HSILoader(),
        rgb_loader=name2dataset_dict[dataset].RGBLoader(),
        svd_loader=name2dataset_dict[dataset].SVDLoader(),
    )
    return o_dataset

