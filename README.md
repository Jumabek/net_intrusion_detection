# net_intrusion_detection_tutorial
Net intrusion detection experiment for DeepLearning class at Inha University. I would be glad you benefit from it as well 

## dataset homepage: https://www.unb.ca/cic/datasets/ids-2017.html


## Accompanying slides
https://docs.google.com/presentation/d/1Rjj1vF0hv8vSJWeDxk23nE4A4w3fv8tBdvsyIBpWTdU/edit?usp=sharing


## 5-Fold CV Results
| Classifier  | 5-Fold Balanced Accuracy |
| ------------- | ------------- |
| Random Forest  | 84.41  |
| Content Linear Softmax  | 80.99  |
| Neural Network with 3 dense layer  | 84.87  |
| Neural Network with 5 dense layer  | 85.57  |
| 1D-CNN with 2conv 1fc layer | 87.11  |

### Softmax
Please run the Softmax.ipynb 

### NN
Please run the NN.ipynb 
There are two NN architectures:
1. 'nn3' - 3 layers
2. 'nn5' - 5 layers

### 1D-CNN
Please run the CNN.ipynb 
There are two 1D-CNN architectures:
1. 'cnn2' - 2 conv layers
2. 'cnn5' - 5 conv layers

