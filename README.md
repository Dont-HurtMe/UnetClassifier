# UnetClassifier
## **model overiew work flow**       

concept Layer : encoder{Resnet50} -> decoder -> **mask** -> flatten{**mask**} -> fully-connected layers{in=2^16,out=3} -> softmax

**detail training & evaluate in create-unet-classifier.ipnyb

train entropy loss vs val entropy loss
<img width="497" alt="image" src="https://github.com/Dont-HurtMe/UnetClassifier/assets/154254885/6093d204-de21-43bf-aa82-1ae644484374">

validate accuracy
<img width="498" alt="image" src="https://github.com/Dont-HurtMe/UnetClassifier/assets/154254885/9a889e62-0d52-4dca-a500-6768e814ee23">

evaluate (test-set)
* entropyloss:__0.7203  
* accuracy:_____0.8815  
* f1-score:_____0.8812 
* precision:____0.8828 
* recall:_______0.8815 
* auc:__________0.9120 

  confusion matrix
  |       || benign | malignant | normal |
  | --- | --- || --- | --- |
  | benign  || 237  | 3 | 2 |
  | malignant  || 4 | 211  | 27  |
  | normal  || 3 Cell  | 47  | 192  |


| Command | Description |
| --- | --- |
| git status | List all new or modified files |
| git diff | Show file differences that haven't been staged |

