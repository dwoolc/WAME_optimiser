# WAME_optimiser


This repo contains the code used to re-create the "Weight–wise Adaptive learning rates with Moving average Estimator" (WAME). The paper can be found at:  
https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2017-50.pdf  

This repo contains a single script for the optimiser itself. The optimiser inherits from the Keras optimizer class, and is intended to be used as a back-prop method for the optimiser parameter in a Keras model.  

Inspiration drawn from the repo from the WAME author at https://github.com/nitbix/keras-oldfork/blob/master/keras/optimizers.py

### Deployment

There are two options for using this method:  

1) Copy the code into the Keras optimiser script within the Keras module. This is a relatively simple extension of the existing methods, and also allows for aliasing of the method alongside other methods. This is how I deploy the method.

2) Importing the Keras Optimiser class into a new script, and then defining the optimiser within that script. In itself, this is less complex as it does not require editing the base Keras code, but does limit the ability to use optimiser aliasing.


