# CSAW-HackML-2020

```bash
├── data 
    └── clean_validation_data.h5 // this is clean data used to evaluate the BadNet and design the backdoor defense
    └── clean_test_data.h5
    └── sunglasses_poisoned_data.h5
├── models
    └── anonymous_bd_net.h5
    └── anonymous_bd_weights.h5
    └── sunglasses_bd_net.h5
    └── sunglasses_bd_weights.h5
├── accuracies
    └── bd1/clean_accuracies_bd1.npy
    └── bd1/poisoned_accuracies_bd1.npy
    └── bd2/cean_accuracies_bd2.npy
├── activations
    └── Activations_bd1.npy
    └── Activations_bd2.npy
```

## I. Dependencies
   1. Python 3.8
   2. Keras 2.4.3
   3. Numpy 1.19.4
   4. H5py 2.10.0
   5. TensorFlow 2.2.0
   6. Kerassurgeon 0.2.0
   
## II. Validation Data
   1. Download the validation and test datasets from [here](https://drive.google.com/drive/folders/13o2ybRJ1BkGUvfmQEeZqDo1kskyFywab?usp=sharing) and store them under `data/` directory.
   2. The dataset contains images from YouTube Aligned Face Dataset. We retrieve 1283 individuals each containing 9 images in the validation dataset.
   3. sunglasses_poisoned_data.h5 contains test images with sunglasses trigger that activates the backdoor for sunglasses_bd_net.h5.

## III. Contents
	1. The files G1.py and G2.py generated repaired models for badnets B1 and B2 respectively. Although both the models implement the same fine-pruning method.
	2. The generated repaired nets are saved in the G1 and G2 folders and can be loaded using Keras.Models.load_model(G2).
	3. The activations folder contains the numpy array of the average activations of each channel in the last convolutional layer which is used to prune the network.
	4. The accuracies folder contains the numpy array with the accuracies of the pruned model w.r.t the no of channels pruned. This is used to plot the graph of the same.
	5. The file calc_activations.py has the function that gets the activations of each layer in the network.

## IV. Instructions
	1. The repaired net can be loaded by using Keras.Models.load_model(G2) for testing on other backdoored data derived from the clean validation data provided to us.
	2. The files G1.py and G2.py generate the repaired networks. 
	3. The files G1.py and G2.py have the code to calculate activations, take the average for each channel and save the array to a file so that this lengthy calculation is not performed again and again. This part is commented out in the code and the arrays are being loaded because we had calculated them earlier. 
	4. The activations are sorted and then used to perform Pruning and Fine-Pruning.
	5. The files G1.py and G2.py also have code for Fine-Tuning, Pruning and Fine Pruning. Make sure to uncomment the relevant part of the code.



## THANK YOU



