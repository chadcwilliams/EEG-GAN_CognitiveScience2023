
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from nn_architecture.ae_networks import TransformerAutoencoder, TransformerDoubleAutoencoder, TransformerFlattenAutoencoder
from helpers.dataloader import Dataloader

# another comment

for s in ['005','010','015','030','060','100']:
    for r in range(5):
        #User input
        data_checkpoint = f'Data/Training Datasets/gansTrialERP_len100_SS{s}_run0{r}.csv'
        ae_checkpoint = f'trained_ae/ae_ddp_1000ep_ss{s}_Run0{r}_enc25.pt'
        print(data_checkpoint)
        print(ae_checkpoint)
        print('')

        #Load
        ae_dict = torch.load(ae_checkpoint, map_location=torch.device('cpu'))
        dataloader = Dataloader(data_checkpoint, col_label='Condition', channel_label='')
        dataset = dataloader.get_data()
        sequence_length = dataset.shape[1] - dataloader.labels.shape[1]

        #Initiate    
        seq_length = sequence_length
        if ae_dict['configuration']['target'] == 'channels':
            model = TransformerAutoencoder(input_dim=ae_dict['configuration']['input_dim'],
                                            output_dim=ae_dict['configuration']['output_dim'],
                                            target=TransformerAutoencoder.TARGET_CHANNELS,
                                            hidden_dim=ae_dict['configuration']['hidden_dim'],
                                            num_layers=ae_dict['configuration']['num_layers'],
                                            num_heads=ae_dict['configuration']['num_heads'],).to('cpu')
        elif ae_dict['configuration']['target'] == 'time':
            model = TransformerAutoencoder(input_dim=seq_length,
                                            output_dim=ae_dict['configuration']['timeseries_out'],
                                            output_dim_2=ae_dict['configuration']['output_dim_2'],
                                            target=TransformerAutoencoder.TARGET_TIMESERIES,
                                            hidden_dim=ae_dict['configuration']['hidden_dim'],
                                            num_layers=ae_dict['configuration']['num_layers'],
                                            num_heads=ae_dict['configuration']['num_heads'], ).to('cpu')
        elif ae_dict['configuration']['target'] == 'full':
            model = TransformerDoubleAutoencoder(input_dim=ae_dict['configuration']['input_dim'],
                                                    output_dim=ae_dict['configuration']['output_dim'],
                                                    sequence_length=seq_length ,
                                                    output_dim_2=ae_dict['configuration']['output_dim_2'],
                                                    hidden_dim=ae_dict['configuration']['hidden_dim'],
                                                    num_layers=ae_dict['configuration']['num_layers'],
                                                    num_heads=ae_dict['configuration']['num_heads'],).to('cpu')
        else:
            raise ValueError(f"Encode target '{ae_dict['configuration']['target']}' not recognized, options are 'channels', 'time', or 'full'.")
        consume_prefix_in_state_dict_if_present(ae_dict['model'],'module.')
        model.load_state_dict(ae_dict['model'])
        model.device = torch.device('cpu')
        print(ae_dict["configuration"]["history"])

        #Test
        plt.figure()
        plt.plot(ae_dict['train_loss'], label='Train Loss')
        plt.plot(ae_dict['test_loss'], label = 'Test Loss')
        plt.title('Losses')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()

        def norm(data):
            return (data-np.min(data)) / (np.max(data) - np.min(data))

        dataset = norm(dataset.detach().numpy())

        fig, axs = plt.subplots(5,1)
        axs[0].set_title(data_checkpoint)
        for i in range(5):
            sample = np.random.choice(len(dataset), 1)
            data = dataset[sample,1:,:]
            axs[i].plot(data[0,:,0], label='Original')
            axs[i].plot(model.decode(model.encode(torch.from_numpy(data)))[0,:,0].detach().numpy(), label='Reconstructed')
            axs[i].legend()
        plt.show()

        os.system("pause")
