

import os
import torch

files = os.listdir('trained_ae')
autoencoder_filenames = [file for file in files if 'ae_ddp_1000ep_2023' in file]

for autoencoder_filename in autoencoder_filenames:
    autoencoder = torch.load(f"trained_ae/{autoencoder_filename}", map_location=torch.device('cpu'))
    filename_details = autoencoder['configuration']['path_dataset'].split('len100')[-1].replace('.csv','')
    torch.save(autoencoder,f'trained_ae/ae_ddp_1000ep{filename_details}_enc25.pt')