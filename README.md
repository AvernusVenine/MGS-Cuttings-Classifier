# MGS Cuttings Classifier

A [ResNet](https://pytorch.org/hub/pytorch_vision_resnet/) model used to automatically classify drill cuttings.

All data needs to be in the form of a singular grain or as sets of 6, as seen in example_photo.jpg.
If data is in that form, it must be divided using the function divide_dataset() found in image_preprocessor.py
