`This repository is highly in devlopment.  
A proper README will follow in the future.`


# SRON-DCGAN
Using deep convolutional generative adversarial networks (DCGANs) for the semantic inpainting of exoplanetary atmosphere spectra

# Introduction 
The ideas and models within this repository are bassed off of ExoGAN, [link](https://github.com/ucl-exoplanets/ExoGAN_public).
...  

# DCGAN
The DCGAN architecture from 'Unsupervised Representational Learning with Deep Convolutional Generative Adversarial Networks (2016)' is used and modified to work with 1x32x32 (Channel, Height, Width) shaped images.  

# Atmospheric Spectra and Parameters Array (ASPA)
Each image contains a Atmospheric Spectra and Parameters Array (ASPA).  
The data used to create the ASPAs (currently) originates from [TauREx](https://github.com/ucl-exoplanets/TauREx_public) and is released within the ExoGAN repository.  
An example spectrum would be as follows.
  
...  
  
The spectrum along with the parameters like planet mass, planet radius, planet temperature and gas mixture ratios are encoded into an ASPA.  
These ASPAs are then used to train the DCGAN.  
After ~8 hours of training on a Tesla M10 using 0.25 % of the original ExoGAN training set, the following result is gathered. 

<p align="center"> <img src="https://github.com/deKeijzer/SRON-DCGAN/blob/master/notebooks/plots/DCGAN_generated_cropped.png?raw=true"> </p>

# Semantic inpainting
...

# Results
...
