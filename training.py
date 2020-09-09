import torch
import os 
import matplotlib.pyplot as plt

from utilities import load_model
from architecture import VAE
from PIL import Image



def run_model(dataloader, args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    network = VAE()
    network.apply(network.initialize_weight)
    network.to(device)

    if args.load:
        load_model(network)
        print("Loaded Trainedmodel!")
    else:
        if not os.path.exists("TrainedModel"):
            os.makedirs("TrainedModel")


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
                        gamma=0.2
                    )

    epoch_history = []

    for epoch in range(args.epochs):
        print(f"Starting Epoch [{epoch+1} / {args.epochs}]")
        
        if not os.path.exists("Output"):
            os.makedirs("Output")

        epoch_loss, epoch_acc = run_epoch(network, optimizer, dataloader, mode=args.mode)

        print("-"*30)
        print(f"Epoch [{epoch + 1: >4}/{args.epochs}] Loss: {epoch_loss:.2e} Acc: {epoch_acc:.2e}")
        print("-"*30)
        
        if args.mode == 'test' or args.mode == 'inference':
            return

        epoch_history.append(epoch_loss)

        if args.mode == 'train':
            lr_scheduler.step()

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
        
    
    if args.mode == 'train':
        torch.save(network.state_dict(), "TrainedModel/finalModel.pth")

    return


def run_epoch(network, optimizer, dataloader, mode):
    device = next(network.parameters()).device

    if mode == 'train':
        network.train()
    else:
        network.eval()

    epoch_loss = 0.0
    iter_count = 1


    if mode == 'train' or mode == 'test':
        for X, y in dataloader:
            if iter_count % 100 == 0:
                print(f"Processing data [{iter_count}/{len(dataloader)}]")
    
            X = X.to(device)
            y = y.to(device)
            loss = 0
            
            if mode == 'train':
                with torch.set_grad_enabled(True):
                    network.zero_grad()
    
                    Z, mean, std = network.encoder(X)
                    X_hat = network.decoder(Z)  # Shape [N, 1, 28, 28]
    
                    rec_loss = torch.sum((X - X_hat)**2, dim=[1,2,3])
                    kl_loss = -0.5 * torch.sum(
                        1 + std - mean**2 - torch.exp(std), 
                        dim=-1)
                    
                    loss = torch.mean(rec_loss + kl_loss)
                    
                    
                    if iter_count == 1:
                        output = X_hat[0].view(28, 28).detach().cpu().numpy()
            
                        fig = plt.figure(figsize=(16, 9))
                        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                        im = plt.imshow(output, cmap='gray')
                        #pos = fig.add_axes([0.93,0.1,0.02,0.35])
                        #fig.colorbar(im, cax=pos)
                        plt.savefig(f"Output/train_{loss : .4f}.png")
                        plt.close(fig)
                    
    
                loss.backward()
                optimizer.step()      
    
    
            elif mode == 'test':
                with torch.no_grad():
                    Z, mean, std = network.encoder(X)
                    X_hat = network.decoder(Z)
    
                    rec_loss = torch.sum((X - X_hat)**2, dim=[1,2,3])
                    kl_loss = -0.5 * torch.sum(
                        1 + std - mean**2 - torch.exp(std), 
                        dim=-1)
                    
                    loss = torch.mean(rec_loss + kl_loss)
                        
    
            #print(f"Loss of {loss:.6f}")
    
            epoch_loss += loss
            iter_count += 1
    
        epoch_loss /= len(dataloader)

    
    if mode == 'inference':
        for i in range(10):
            with torch.no_grad():
                xi = torch.normal(torch.zeros([1, network.hidden_dimension]))
                xi = xi.to(device)
                output = network.decoder(xi).view(28, 28).cpu().numpy()
                # output = Image.fromarray(out_img.view(3, H, W).cpu().numpy())

            fig = plt.figure(figsize=(16, 9))
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            im = plt.imshow(output, cmap='gray')
            #pos = fig.add_axes([0.93,0.1,0.02,0.35])
            #fig.colorbar(im, cax=pos)
            plt.savefig(f"Output/{i : 02d}.png")
            plt.close(fig)


    return epoch_loss, 0






