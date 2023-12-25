# UnetClassifier
## **model overiew work flow**       

concept Layer : encoder{Resnet50} -> decoder -> **mask** -> flatten{**mask**} -> fully-connected layers{in=2^16,out=3} -> softmax



<img width="960" alt="image" src="https://github.com/Dont-HurtMe/UnetClassifier/assets/154254885/b711ed1a-0d66-41e9-88d8-b842b1f1e6e4">





**log validate accuracy**


<img width="504" alt="image" src="https://github.com/Dont-HurtMe/UnetClassifier/assets/154254885/c7a4aabb-041d-4baa-9ed1-11a254fb85ae">




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

* detail training & evaluate in create-unet-classifier.ipnyb
* output of UnetClassifier in example-output-model.ipynb



**detail OOB**
________________________________________

1. DataLoad.py --> generate data from dataframe (pattern columns in dataframe : [image_path,mask,original_label,encoder_label]) transform and torch.dataLoad sample use in In[3] from 2create-unet-classifier.ipynb


2. __model__.py --> load pretrain unet segment and freeze parameter then connect fully-layers 


3. train.py --> fit (optimize weight fully layer only) & save model (func. from __tool__.py) return model , loss_train , loss_val , loss_acc 


4. __out__.py --> this object is used for predict and output UnetClassifier model sample from In[2],In[13] example-output-model.ipynb


5. __tool__.py --> this object is used to hold all the function needed for this project.such as polygons (for draw mask unet output) , save model , trainloop , validateloop and transform resize images









