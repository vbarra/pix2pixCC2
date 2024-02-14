
# Materials for the paper:

 

*An experimental study on EUV-to-magnetogram image translation using conditional Generative Adversarial Networks*, M Dannehl, V Delouille, V Barra

 

Submitted to Earth and Space Science

 

## Modifications in pix2pixCC

Part of the pix2pixCC code have been modified to optimize and structure the training and testing of the model under different configurations.

- the configuration of parameters is defined in  pix2pixCC/config.yaml

- The time stamps of training, validation and test files are generated in    pix2pixCC2/relevant_timestamps_generator.py

- pix2pixCC/solar_dataset.py then prepares the AIA and HMI datasets, according to the relevant_timestamps

- pix2pixCC/pix2pixCC_Train.py and  pix2pixCC/pix2pixCC_Test.py performing training using the various timestamps defined

 

## Analysis of the results

Several scripts provides analysis of input data (histogram) and of results. In particular,

    pix2pixCC2/results_compare_npz_with_ssim.py compare the results from different trained models.
