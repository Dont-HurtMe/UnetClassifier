# UnetClassifier
##### model overiew work flow
------------------------------------------------------------------------------------------------------------------------------------
##### concept Model
##### { input image(256,256) ---> predict **mask** by encoder to decoder layers ---> predict **label** by output(mask) of unet ---> return **mask** and **label }
-------------------------------------------------------------------------------------------------------------------------------------                   
##### concept Layer : encoder{Resnet50} -> decoder --> **mask** --> flatten{**mask**} --> fully-connected layers{in=2^16,out=3} --> softmax
