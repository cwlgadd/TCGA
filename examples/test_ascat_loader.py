import torch
import numpy as np

import os
import TCGA


def debug_loader():
    # TODO: turn into unit test

    data_module = TCGA.data_modules.ascat.ASCATDataModule(wgd=['0'], cancer_types=['ACC', 'BRCA'])
    print(data_module)
    print(data_module.weight_dict)
    print(data_module.label_encoder.classes_)

    loader_list = {'train': data_module.train_dataloader(),
                   'test': data_module.test_dataloader(),
                   'validation': data_module.val_dataloader(),
                   }
    for key in loader_list:
        print(f'\n{key} set\n=============')
        for batch_idx, batch in enumerate(loader_list[key]):
            print(f'\nBatch {key} index {batch_idx}')
            print(f'Batch {batch.keys()}')
            print(f"Feature shape {batch['feature'].shape}, label shape {batch['label'].shape}")
            print(f"label counts {torch.unique(batch['label'], return_counts=True)}")
            print(np.unique(batch['feature'], axis=0).shape)
            # for s in range(batch['feature'].shape[0]):
            #     print(f"input  {batch['feature'][s, 0, 0, :]}")   # N x length x regions x num_sequences
            #     print(f"output  {batch['label'][s]}")


if __name__ == '__main__':

    debug_loader()