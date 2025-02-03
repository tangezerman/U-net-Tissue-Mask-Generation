# U-Net Tissue Mask Generation with VGG16 Backbone

The model was trained using a private dataset that was provided to students in the class. The dataset has over 290 GBs worth of high resolution WSI, coloured by various staining techniques. Our goal was to seperate the tissues from background and obtain tissue masks using a deep learning approach of our choice, essentialy semantic binary segmentation. 
The model architecture uses weights of VGG 16 trained on IMAGENET. 

The trained model is under Models directory. You can extract it into the same directory for testing purposes.

When creating patches, `multithreaded_patches.py` utilizes multiple parallel python processes to speed up the process.

![Training history][history]

Training took 8 epochs with a batch size of 48 on a GTX 1070 on pytorch. 

## Sample predictions for patch size 256
![sample_predictions_256_48_0.png][pred1]
![sample_predictions_256_48_0.png][pred2]

## Validation set results

![Results][def]


[history]: image-1.png
[pred1]: sample_predictions_256_48_0.png
[pred2]: pred.png
[def]: image-2.png