# Deep Learning based network intrusion detection in PyTorch

Net intrusion detection experiment for Final Project of DeepLearning class (2019) at Inha University.

## Dataset homepage: https://www.unb.ca/cic/datasets/ids-2017.html

## Contributions

1. Correct Evaluation Metric
2. Adressing data imblance
3. Benchmark results for different ML models
4. Running code for training/evaluating

## Accompanying slides

https://docs.google.com/presentation/d/1Rjj1vF0hv8vSJWeDxk23nE4A4w3fv8tBdvsyIBpWTdU/edit?usp=sharing

## Model Performance using K-Fold Cross-Validation

| Classifier                        | 5-Fold Balanced Accuracy |
| --------------------------------- | ------------------------ |
| Content Linear Softmax            | 76.27                    |
| Neural Network with 3 dense layer | 85.73                    |
| Neural Network with 5 dense layer | 85.63                    |
| 1D-CNN with 2conv 1fc layer       | 87.13                    |
| CNN with 5conv layer              | 87.16                    |
| Random Forest                     | 80.09                    |

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
