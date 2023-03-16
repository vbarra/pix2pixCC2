"""
Train code for the pix2pixCC model

"""

#==============================================================================

if __name__ == '__main__':

    #--------------------------------------------------------------------------
    import os
    import datetime  
    import numpy as np
    from tqdm import tqdm
     
    import torch
    from torch.utils.data import DataLoader
    
    from pix2pixCC_Networks import Discriminator, Generator, Loss
    from pix2pixCC_Options import Option
    #from pix2pixCC_Pipeline import CustomDataset
    from pix2pixCC_Utils import Manager, weights_init
    
    from solar_dataset import SolarDataset
    import shutil
    import tensorboardX
    
    #--------------------------------------------------------------------------
    
    start_time = datetime.datetime.now()
    start_time_str = start_time.strftime("%Y%m%d_%H%M")

    #--------------------------------------------------------------------------
    # [1] Initial Conditions Setup
    
    torch.backends.cudnn.benchmark = False

    config_file = 'config.yaml'
    opt = Option(config_file, start_time_str).parse()
    
    shutil.copy(config_file, os.path.join(opt.result_dir, config_file))
    
    train_writer = tensorboardX.SummaryWriter(os.path.join(opt.result_dir,'logs'))
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_ids)
    
    print("CPU count is: " +str(os.cpu_count()))

    device = torch.device('cuda:0')
    dtype = torch.float16 if opt.data_type == 16 else torch.float32
    
    lr = opt.lr
    batch_sz = opt.batch_size
    
    # --- Dataset upload ---
    dataset = SolarDataset(opt, opt.relevant_timestamp_path_train)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_sz,
                             num_workers=opt.n_workers,
                             shuffle=not opt.no_shuffle)
    
    # --- Network and Optimizer update ---
    G = torch.nn.DataParallel(Generator(opt)).apply(weights_init).to(device=device, dtype=dtype)
    D = torch.nn.DataParallel(Discriminator(opt)).apply(weights_init).to(device=device, dtype=dtype)

    G_optim = torch.optim.Adam(G.parameters(), lr=lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)
    D_optim = torch.optim.Adam(D.parameters(), lr=lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)
    criterion = Loss(opt)
    
    # --- Resume check ---
    G_init_path = opt.model_dir + '/' + str(opt.latest_iter) + '_G.pt'
    D_init_path = opt.model_dir + '/' + str(opt.latest_iter) + '_D.pt'
    if os.path.isfile(G_init_path) and os.path.isfile(D_init_path) :
        init_iter = opt.latest_iter
        print("Resume at iteration: ", init_iter)
        
        G.module.load_state_dict(torch.load(G_init_path))
        D.module.load_state_dict(torch.load(D_init_path))

        init_epoch = int(float(init_iter)/(batch_sz*len(data_loader)))
        current_step = int(init_iter)

    else:
        init_epoch = 1
        current_step = 0
   

    manager = Manager(opt, train_writer)    
    
    #--------------------------------------------------------------------------
    # [2] Model training
    #from  torchsummary import summary
    #print(summary(G, input_size=(opt.input_ch,opt.data_size,opt.data_size), batch_size=opt.batch_size))
    
    dataset_length = len(data_loader)
    total_step = opt.n_epochs * dataset_length * batch_sz
    
    best_val_loss_G = np.inf
    failed_improvement_counter = 0
    patience = opt.patience

    for epoch in range(init_epoch, opt.n_epochs + 1):
        
        G.train()
        D.train()
        
        train_losses_G = []
        train_losses_D = []
        
        for input, target, timestamps in tqdm(data_loader, disable=False):
            
            current_step += batch_sz
            
            input, target = input.to(device=device, dtype=dtype), target.to(device, dtype=dtype)
            
            D_loss, G_loss, target_tensor, generated_tensor = criterion(D, G, input, target)

            G_optim.zero_grad()
            G_loss.backward()
            G_optim.step()

            D_optim.zero_grad()
            D_loss.backward()
            D_optim.step()
            
            train_losses_G.append(G_loss.detach().item())
            train_losses_D.append(D_loss.detach().item())

        train_loss_G = np.mean(train_losses_G)
        train_loss_D = np.mean(train_losses_D)

        package = {'Epoch': epoch,
                   'D_loss': train_loss_D,
                   'G_loss': train_loss_G,
                   'D_state_dict': D.module.state_dict(),
                   'G_state_dict': G.module.state_dict(),
                   'D_optim_state_dict': D_optim.state_dict(),
                   'G_optim_state_dict': G_optim.state_dict(),
                   'input_tensor': input,
                   'target_tensor': target_tensor,
                   'generated_tensor': generated_tensor.detach(),
                   'timestamps': timestamps
                   }

        manager(package)
            
    #--------------------------------------------------------------------------
    # [3] Model Checking 
            
        # validation and early stopping
        if opt.val_freq and epoch % opt.val_freq == 0:

            G.eval()
            D.eval()
            
            for p in G.parameters():
                p.requires_grad_(False)
            
            for p in D.parameters():
                p.requires_grad_(False)
            
            test_dataset = SolarDataset(opt, opt.relevant_timestamp_path_test)
            test_data_loader = DataLoader(dataset=test_dataset,
                                          batch_size=opt.batch_size,
                                          num_workers=opt.n_workers,
                                          shuffle=False)  
            val_losses_G = []     
            val_losses_D = []    

            for input, target, timestamps in tqdm(test_data_loader):
                input, target = input.to(device=device, dtype=dtype), target.to(device, dtype=dtype)
                #fake = G(input)
                
                D_loss, G_loss, target_tensor, fake = criterion(D, G, input, target)
                
                val_losses_G.append(G_loss.detach().item())
                val_losses_D.append(D_loss.detach().item())
            
            print("Building mean of the test losses of all test batches")
            val_loss_G = np.mean(val_losses_G)
            val_loss_D = np.mean(val_losses_D)
            
            print("Logging test losses")
            
            train_writer.add_scalar('G_loss_val', val_loss_G, epoch)
            train_writer.add_scalar('D_loss_val', val_loss_D, epoch)
            
            print('Creating plot of test img')
            
            # plot image
            path = os.path.join(opt.image_dir_test, "E"+str(epoch) + '_' + 'all.png')
            manager.save_aia_hmi(
                path,
                input,
                target,
                fake,
                timestamps,
                epoch=epoch)
                
            ## early stopping and save checkpoints
            if val_loss_G < best_val_loss_G:
                best_val_loss_G = val_loss_G
                failed_improvement_counter = 0
                
                #save checkpoint
                package = {'Epoch': epoch, 
                           'D_state_dict': D.module.state_dict(),
                           'G_state_dict': G.module.state_dict()}
                manager.save(package, model=True, suffix='_best')
                
            else:
                failed_improvement_counter += opt.val_freq
                if failed_improvement_counter >= patience:
                    print('Canceled training (early stopping) after epoch' +str(epoch))
                    break # end training
                    

            for p in G.parameters():
                p.requires_grad_(True)
            
            for p in D.parameters():
                p.requires_grad_(True)


    #--------------------------------------------------------------------------    
    
    # save last checkpoint
    package = {'Epoch': epoch, 
               'D_state_dict': D.module.state_dict(),
               'G_state_dict': G.module.state_dict()}
    manager.save(package, model=True, suffix='_last')
    
    #--------------------------------------------------------------------------    
    
    end_time = datetime.datetime.now()
    
    print("Total time taken: ", end_time - start_time)
    
    train_writer.flush()


#==============================================================================
