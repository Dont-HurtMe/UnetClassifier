# UnetClassifier
## **model overiew work flow**       

concept Layer : encoder{Resnet50} -> decoder -> **mask** -> flatten{**mask**} -> fully-connected layers{in=2^16,out=3} -> softmax

**log validate accuracy**


<img width="497" alt="image" src="https://github.com/Dont-HurtMe/UnetClassifier/assets/154254885/9a889e62-0d52-4dca-a500-6768e814ee23">



**evaluate (test-set)**
* entropyloss: _______0.7264 
* accuracy: __________0.9242 
* f1-score: __________0.9236
* precision: _________0.9245 
* recall: _____________0.9242
* auc: _______________0.9509

**confusion matrix**
| label name| benign | malignant | normal |
| --- | --- | --- | --- |
| benign  | 241  | 1 | 0 |
| malignant  | 7 | 221  | 14  |
| normal  | 7 | 26 | 209  |

**detail training & evaluate in create-unet-classifier.ipnyb






