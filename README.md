# Repainting the past
DockerCon EU 2017 project

## Running the custom network
To run the custom network written in C++ and OpenCL, open the project in Xcode and hit run. That easy. The project will then proceed to begin training via 'overfitting' on a single image to prove the loss function decreases over iterations, however this can easily be changed to train on numerous different images.
 
## Running the tensorflow network
In order to run the tensorflow network, switch to the tf_network branch, and run `python network_dist_train.py`. This will train the network using all the images stored in `./data`. (this folder needs to be filled first)

Upon running, the network will train, storing data in the folder `./train_dir`. (in some cases this folder will not be created automatically so just create a new folder if this happens)

Once the network is trained, store some images in a folder named `./out`, and then run `python network_eval.py`.
