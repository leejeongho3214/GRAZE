import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from constants.h2o_constants import h2o_obj_name
from constants.hot3d_constants import hot3d_obj_name
from constants.grab_constants import grab_obj_name
from lib.datasets.h2o import (
     MotionHOT3D, ContactHOT3D
)


def _resolve_seed(config):
    texthom_config = getattr(config, "texthom", None)
    if texthom_config is not None and hasattr(texthom_config, "exp"):
        return int(getattr(texthom_config.exp, "seed", 0))
    return int(getattr(config, "seed", 0))


def get_dataset(dataset_name, dataset_config, test_flag):
    assert dataset_name in [
        "Sequenceh2o", "Contacth2o", "Contacthot3d", "Motionh2o", "Motionhot3d",
        "Sequencegrab", "Contactgrab", "Motiongrab", 
        "Sequencearctic", "Contactarctic", "Motionarctic", "Sequencehot3d"
    ]
    dataset = MotionHOT3D(test_flag=test_flag, **dataset_config)
    
    return dataset

def get_dataloader(dataset_name, config, data_config, test=False, shuffle=False):
    dataset = get_dataset(
        dataset_name, 
        data_config, 
        test
    )
    seed = _resolve_seed(config)
    generator = torch.Generator()
    generator.manual_seed(seed)

    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    if config.balance_weights and not test:
        sampler = WeightedRandomSampler(
            torch.FloatTensor(dataset.balance_weights),
            len(dataset),
            generator=generator,
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=config.batch_size, 
            sampler=sampler,
            num_workers=config.num_workers,
            drop_last=config.drop_last,
            worker_init_fn=seed_worker,
            generator=generator,
        )
    else:
        dataloader = DataLoader(
            dataset, 
            batch_size=config.batch_size, 
            shuffle=bool(shuffle),
            num_workers=config.num_workers,
            worker_init_fn=seed_worker,
            generator=generator,
        )
        
    return dataset, dataloader
