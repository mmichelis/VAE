# ------------------------------------------------------------------------------
# Training the network defined in architecture using the data loaded using data-
# loader.
# Also provides the test and inference modes on already trained models.
# ------------------------------------------------------------------------------

import torch
import os 
import matplotlib.pyplot as plt

from utilities import load_model
from architecture import VAE
from PIL import Image



def run_model(dataloader, args):
    """
    Run a model in either training mode or testing mode. If the model exists, then the weights are loaded from this trained model.

    Arguments:
        dataloader (torch.utils.data.DataLoader) : contains the data used for training/testing.
        args (argparse.ArgumentParser) : command line arguments
        
    Returns:
        None
    """
    # Load the device on which we're running the model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ### Initialize network
    network = VAE()
    network.apply(network.initialize_weight)
    network.to(device)

    if args.load:
        load_model(network)
    else:
        if not os.path.exists("TrainedModel"):
            os.makedirs("TrainedModel")

    # Output directory might not exist yet
    if not os.path.exists("Output"):
        os.makedirs("Output")


    if args.mode == 'train':
        ### Create optimizer and scheduler

        # optimizer = torch.optim.SGD(
        #                     network.parameters(), 
        #                     lr=5e-3, 
        #                     momentum=0.9, 
        #                     weight_decay=1e-4
        #                 )
        
        optimizer = torch.optim.Adam(
                            network.parameters(), 
                            lr=5e-3,
                            weight_decay=1e-4
                        )
        

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                            optimizer, 
                            milestones=[int(args.epochs/3), int(2*args.epochs/3)], 
                            gamma=0.5
                        )

        epoch_history = []

        ### Run epochs
        for epoch in range(args.epochs):
            print(f"Starting Epoch [{epoch+1} / {args.epochs}]")

            epoch_loss, epoch_acc = run_epoch(network, optimizer, dataloader, mode=args.mode)

            ### Print statistics
            print("-"*30)
            print(f"Epoch [{epoch + 1: >4}/{args.epochs}] Loss: {epoch_loss:.2e} Acc: {epoch_acc:.2e}")
            print("-"*30)

            ### Update the learning rate based on scheduler
            lr_scheduler.step()
            
            epoch_history.append(epoch_loss)

            ### Store model as backup
            if epoch % args.checkpoint_epochs == 0:
                torch.save(network.state_dict(), "TrainedModel/modelBackup.pth")

                ### Create and Store Plots
                plt.figure(figsize=(12,9))
                plt.plot(epoch_history, label='Loss History')

                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.xlim(0, epoch)
                plt.legend()
                plt.grid(True)
                plt.savefig("TrainedModel/loss_plot.png", bbox_inches='tight')
            
        ### Save final model
        torch.save(network.state_dict(), "TrainedModel/finalModel.pth")


    elif args.mode == 'test':
        epoch_loss, epoch_acc = run_epoch(network, optimizer, dataloader, mode=args.mode)

        ### Print statistics
        print("-"*30)
        print(f"Loss: {epoch_loss:.2e} Acc: {epoch_acc:.2e}")
        print("-"*30)
            


    elif args.mode == 'inference':
        _, _ = run_epoch(network, optimizer, dataloader, mode=args.mode)


    return


def run_epoch(network, optimizer, dataloader, mode):
    """
    Run one epoch of training or testing.
    
    Arguments:
        network (nn.Sequential) : the network model.
        optimizer (torch.optim) : optimization algorithm for the model.
        dataloader (torch.utils.data.DataLoader): contains the data used for training/testing.
        mode (String) : train, test or inference
        
    Returns:
        Loss and Acc (currently not implemented) in this epoch. Loss only relevant for training, Acc only relevant for validation.
    """
    # Get the device based on the model
    device = next(network.parameters()).device

    if mode == 'train':
        network.train()
    else:
        network.eval()

    epoch_loss = 0.0
    iter_count = 1


    if mode == 'train':
        ### Iterate over data
        for X, y in dataloader:
            # Print n update messages per epoch
            if iter_count % int(len(dataloader)/5) == 0:
                print(f"Processing data [{iter_count}/{len(dataloader)}]")
    
            X = X.to(device)
            y = y.to(device)
            loss = 0
            
            with torch.set_grad_enabled(True):
                network.zero_grad()

                Z, mean, std = network.encoder(X)
                X_hat = network.decoder(Z)  # Shape [N, 1, 28, 28]

                # Compute reconstruction loss between input and output of network
                rec_loss = torch.sum((X - X_hat)**2, dim=[1,2,3])
                
                # Compute Kullback-Leibler divergence 
                kl_loss = -0.5 * torch.sum(
                    1 + std - mean**2 - torch.exp(std), 
                    dim=-1)
                
                loss = torch.mean(rec_loss + kl_loss)

            loss.backward()
            optimizer.step()   
                      
            # Store the first output in every epoch
            if iter_count == 1:
                output = X_hat[0].view(28, 28).detach().cpu().numpy()
    
                fig = plt.figure(figsize=(16, 9))
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                im = plt.imshow(output, cmap='gray')
                plt.savefig(f"Output/train_{loss : .4f}.png")
                plt.close(fig)
                   
            #print(f"Loss of {loss:.6f}")
    
            epoch_loss += loss
            iter_count += 1
    
        epoch_loss /= len(dataloader)


    elif mode == 'test':
        ### Iterate over data
        for X, y in dataloader:
            # Print n update messages per epoch
            if iter_count % int(len(dataloader)/5) == 0:
                print(f"Processing data [{iter_count}/{len(dataloader)}]")
    
            X = X.to(device)
            y = y.to(device)
            loss = 0
            
            with torch.no_grad():
                Z, mean, std = network.encoder(X)
                X_hat = network.decoder(Z)
                
                # Compute reconstruction loss between input and output of network
                rec_loss = torch.sum((X - X_hat)**2, dim=[1,2,3])
                
                # Compute Kullback-Leibler divergence 
                kl_loss = -0.5 * torch.sum(
                    1 + std - mean**2 - torch.exp(std), 
                    dim=-1)
                
                loss = torch.mean(rec_loss + kl_loss)

            # Store the first output in every epoch
            if iter_count == 1:
                output = X_hat[0].view(28, 28).detach().cpu().numpy()
    
                fig = plt.figure(figsize=(16, 9))
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                im = plt.imshow(output, cmap='gray')
                plt.savefig(f"Output/train_{loss : .4f}.png")
                plt.close(fig)

            #print(f"Loss of {loss:.6f}")
    
            epoch_loss += loss
            iter_count += 1
    
        epoch_loss /= len(dataloader)

    
    elif mode == 'inference':
        for i in range(10):
            with torch.no_grad():
                # Generate a standard normal gaussian random vector to feed into network.
                xi = torch.normal(torch.zeros([1, network.hidden_dimension]))
                xi = xi.to(device)
                output = network.decoder(xi).view(28, 28).cpu().numpy()

            fig = plt.figure(figsize=(16, 9))
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            im = plt.imshow(output, cmap='gray')
            plt.savefig(f"Output/{i : 02d}.png")
            plt.close(fig)


    return epoch_loss, 0






