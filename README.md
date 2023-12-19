# UnetClassifier
## **model overiew work flow**       

concept Layer : encoder{Resnet50} -> decoder -> **mask** -> flatten{**mask**} -> fully-connected layers{in=2^16,out=3} -> softmax

**log validate accuracy**


<img width="497" alt="image" src="https://github.com/Dont-HurtMe/UnetClassifier/assets/154254885/9a889e62-0d52-4dca-a500-6768e814ee23">



**evaluate (test-set)**
* entropyloss:__0.7203  
* accuracy:_____0.8815  
* f1-score:_____0.8812 
* precision:____0.8828 
* recall:________0.8815 
* auc:__________0.9120 

**confusion matrix**
| label name| benign | malignant | normal |
| --- | --- | --- | --- |
| benign  | 237  | 3 | 2 |
| malignant  | 4 | 211  | 27  |
| normal  | 3 | 47  | 192  |

**detail training & evaluate in create-unet-classifier.ipnyb






