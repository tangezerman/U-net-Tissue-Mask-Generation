# U-Net Tissue Mask Generation with VGG16 Backbone

The model was trained using a private dataset that was provided to students in the class. The dataset has over 290 GBs worth of high resolution WSI, coloured by various staining techniques. Our goal was to seperate the tissues from background and obtain tissue masks using a deep learning approach of our choice, essentialy semantic binary segmentation. 
The model architecture uses weights of VGG 16 trained on IMAGENET. 

The trained model is under Models directory. You can extract it into the same directory for testing purposes.