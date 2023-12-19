# UnetClassifier
model overiew work flow
-------------------------------------------------------------------------------------------------------------------------------------                   
concept Layer : encoder{Resnet50} -> decoder --> **mask** --> flatten{**mask**} --> fully-connected layers{in=2^16,out=3} --> softmax
