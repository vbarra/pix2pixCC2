"""
Test code for the pix2pixCC model

"""

#==============================================================================
# [1] Initial Conditions Setup

if __name__ == '__main__':
   
    #--------------------------------------------------------------------------
    import os
    import sys
    import datetime
    import numpy as np
    from glob import glob
    from tqdm import tqdm
    
    import torch
    from torch.utils.data import DataLoader
    
    from pix2pixCC_Options import Option
    #from pix2pixCC_Pipeline import CustomDataset
    from pix2pixCC_Networks import Generator
    from pix2pixCC_Utils import Manager

    from solar_dataset import SolarDataset

    #--------------------------------------------------------------------------

    ## This option determines for which run the testing will be done, if the 
    ## option is not present, the hardcoded timestamp will be used instead
    ## this option cannot be defined in pix2pixCC_Options.py because it is 
    ## used to know where the according config file of the selected run is 
    ## stored.
    test_run_timestamp_key = "--test_run_timestamp=" # e.g. "20220509_1154"
    if len(sys.argv) > 1 and test_run_timestamp_key in sys.argv[1]:
        start_time_str = sys.argv[1].replace(test_run_timestamp_key, "")
        del sys.argv[1] # delete the option so pix2pixCC_Options.py wont be confused
    else:
        # select a run by hardcoding and not providing the command line option
        start_time_str = datetime.datetime(2022,5,9, 11,54).strftime("%Y%m%d_%H%M")

    print('Start of testing for the selected run: ', start_time_str)
    #--------------------------------------------------------------------------
    torch.backends.cudnn.benchmark = True
    
    config_file = "config.yaml" # common file name of the config files
    ## load dummy options - only to automatically calculate the path to the config file of the selected run
    dummy_opt = Option(config_file, start_time_str).parse()

    ## load the config of the selected run
    opt = Option(os.path.join(dummy_opt.result_dir, config_file), start_time_str).parse()
    
    #os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_ids)
    device = torch.device('cuda:0')
    
    STD = opt.dataset_name
    #opt.relevant_timestamp_path_test = '../relevant_timestamps_single.txt' # override manually if needed
    dataset = SolarDataset(opt, opt.relevant_timestamp_path_test)
    
    if opt.n_test_exports == 0:
        batch_size = 1
    else:
        batch_size = int(np.ceil(dataset.__len__() / opt.n_test_exports))

    test_data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=False)
    iters = opt.iteration
    #step = opt.save_freq
    
    
    #--------------------------------------------------------------------------
    
    if (iters == False) or (iters == -1) :
        
        dir_model = opt.model_dir + '/*_G.pt'
        ITERATIONs = sorted([int(os.path.basename(x)[1:].split('_')[0]) for x in glob(dir_model)])
        
        if len(ITERATIONs)==0:
            raise RuntimeError('No data found for run started at ' + start_time_str)
        
        for ITERATION in ITERATIONs:
            
            path_model = opt.model_dir + '/E{}_G.pt'.format(str(ITERATION))
            dir_image_save = opt.image_dir_test + '/E{}'.format(str(ITERATION))
            
            if not os.path.isfile(path_model):                                               
                path_model = opt.model_dir + '/E{}_last_G.pt'.format(str(ITERATION))
                if not os.path.isfile(path_model):
                    if opt.skip_best_checkpoints:  
                        continue
                    else:                                             
                        path_model = opt.model_dir + '/E{}_best_G.pt'.format(str(ITERATION))
                    
            if opt.skip_existing_folders and os.path.isdir(dir_image_save) == True:
                print("Skipping", dir_image_save, "because folder already exists.")
                pass
            else:                                             
                os.makedirs(dir_image_save, exist_ok=True)

                G = torch.nn.DataParallel(Generator(opt)).to(device)
                G.module.load_state_dict(torch.load(path_model))

                manager = Manager(opt)

                with torch.no_grad():
                    G.eval()
                    for input, target, timestamps in tqdm(test_data_loader):
                        input = input.to(device)

                        fake = G(input)

                        #UpIB = opt.saturation_upper_limit_target
                        #LoIB = opt.saturation_lower_limit_target

                        # np_fake = fake.cpu().numpy().squeeze() #*((UpIB - LoIB)/2) +(UpIB+ LoIB)/2

                        #  #--------------------------------------
                        # if len(np_fake.shape) == 3:
                        #     np_fake = np_fake.transpose(1, 2 ,0)

                        # #--------------------------------------
                        # if opt.logscale_target == True:
                        #     np_fake = 10**(np_fake)

                        #--------------------------------------
                        # if opt.data_format_input in ["tif", "tiff"]:
                        #     pil_image = Image.fromarray(np_fake)
                        #     pil_image.save(os.path.join(dir_image_save, target[0] + '_AI.fits'))
                        # elif opt.data_format_input in ["npy"]:
                        #     np.save(os.path.join(dir_image_save, target[0] + '_AI.fits'), np_fake, allow_pickle=True)
                        # elif opt.data_format_input in ["fits", "fts"]:       
                        #     fits.writeto(os.path.join(dir_image_save, target[0] + '_AI.fits'), np_fake)
                        # else:
                        #     NotImplementedError("Please check data_format_target option. It has to be fit or npy or fits.")

                        if opt.export_plots:
                            path = os.path.join(dir_image_save, 'E' + str(ITERATION) + "_" + timestamps[0] +'_test.png')
                            manager.save_aia_hmi(
                                path,
                                input,
                                target,
                                fake,
                                timestamps)
                            
                        if opt.export_npz_fake:
                            path = os.path.join(dir_image_save, "HMI" + timestamps[0] +'_fake.npz')
                            manager.save_hmi_npz(path, fake[0])
                        if opt.export_npz_target:
                            path = os.path.join(dir_image_save, "HMI" + timestamps[0] +'_target.npz')
                            manager.save_hmi_npz(path, target[0])
#    --------------------------------------------------------------------------
    
    else:
        ITERATION = int(iters)
        path_model = opt.model_dir + '/E{}_G.pt'.format(str(ITERATION))
        dir_image_save = opt.image_dir_test + '/E{}'.format(str(ITERATION))
        os.makedirs(dir_image_save, exist_ok=True)
        
        if not os.path.isfile(path_model):                                               
                path_model = opt.model_dir + '/E{}_last_G.pt'.format(str(ITERATION))
                if not os.path.isfile(path_model):
                    if opt.skip_best_checkpoints:  
                        raise RuntimeError(path_model + " does not exist.")
                    else:                                             
                        path_model = opt.model_dir + '/E{}_best_G.pt'.format(str(ITERATION))
        
        if opt.skip_existing_folders and os.path.isdir(dir_image_save) == True:
            print("Skipping", dir_image_save, "because folder already exists.")
            pass
        else: 
            G = torch.nn.DataParallel(Generator(opt)).to(device)
            G.module.load_state_dict(torch.load(path_model))
            
            manager = Manager(opt)
            
            with torch.no_grad():
                G.eval()
                for input,  target, timestamp in tqdm(test_data_loader):
                    input = input.to(device)
                    fake = G(input)
                    
                    # UpIB = opt.saturation_upper_limit_target
                    # LoIB = opt.saturation_lower_limit_target
                    
                    # np_fake = fake.cpu().numpy().squeeze() *((UpIB - LoIB)/2) +(UpIB+ LoIB)/2
                    
                    # #--------------------------------------
                    # if len(np_fake.shape) == 3:
                    #     np_fake = np_fake.transpose(1, 2 ,0)
                    
                    # #--------------------------------------
                    # if opt.logscale_target == True:
                    #     np_fake = 10**(np_fake)
                    
                    # if opt.save_scale != 1:
                    #     np_fake = np_fake*np.float(opt.save_scale)
                    
                    # #--------------------------------------
                    # if opt.data_format_input in ["tif", "tiff"]:
                    #     pil_image = Image.fromarray(np_fake)
                    #     pil_image.save(os.path.join(dir_image_save, target[0] + '_AI.fits'))
                    # elif opt.data_format_input in ["npy"]:
                    #     np.save(os.path.join(dir_image_save, target[0] + '_AI.fits'), np_fake, allow_pickle=True)
                    # elif opt.data_format_input in ["fits", "fts"]:       
                    #     fits.writeto(os.path.join(dir_image_save, target[0] + '_AI.fits'), np_fake)
                    # else:
                    #     NotImplementedError("Please check data_format_target option. It has to be fit or npy or fits.")
                     
                    if opt.export_plots:
                        path = os.path.join(dir_image_save, 'E' + str(ITERATION) + "_" + timestamps[0] +'_test.png')
                        manager.save_aia_hmi(
                            path,
                            input,
                            target,
                            fake,
                            timestamps)
                    
                    if opt.export_npz:
                        path = os.path.join(dir_image_save, "HMI" + timestamps[0] +'.npz')
                        manager.save_hmi_npz(path, fake[0])
                    
#==============================================================================
