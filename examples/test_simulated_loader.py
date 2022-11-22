import torch
import numpy as np
import TCGA


def debug_loader(plot=False, length=20):

    data_module = TCGA.data_modules.simulated.MarkovDataModule(classes=2,
                                                               steps=1,
                                                               n=1000,
                                                               length=length,
                                                               n_class_bases=2,
                                                               n_bases_shared=0)
    print(data_module)

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

            if plot:
                import matplotlib.pyplot as plt
                strand, channel = 0, 0
                for l in [0, 1]:
                    samples = batch['feature'][batch['label'] == l, :][:, strand, channel, :]
                    for s in range(samples.shape[0]):
                        plt.scatter(np.linspace(1, length, length), samples[s, :] + 0.01*np.random.randn(samples[s,:].shape[0]))
                    plt.title(f'Label {l}')
                plt.show()


if __name__ == '__main__':

    debug_loader()
