# tf-image-augmentations
TF compatible image augmentations
All of the functions defined here are compatible as mapping functions for tf.data.Dataset.map

The following modules are included:

1. affine - affine transformations
2. elastic - elastic deformation (inspired by the U-Net paper) 
3. binary mask - extract information from an object binary mask
4. seg_aug - segmentation augmentations, random augmentations that preserve compatibility between an input image and it's segmentation mask
5. utils - general utils, mostly involving dimensions.


If you're using my work for anything other than personal use remember to give credit to Sol Yarkoni.
 
