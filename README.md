# deep learning based network intrusion detection in PyTorch

Net intrusion detection experiment for Final Project of DeepLearning class at Inha University.

## dataset homepage: https://www.unb.ca/cic/datasets/ids-2017.html

## Contributions

1. Correct Evaluation Metric
2. Adressing data imblance
3. Benchmark results for different ML models
4. Running code for training/evaluating

## Accompanying slides

https://docs.google.com/presentation/d/1Rjj1vF0hv8vSJWeDxk23nE4A4w3fv8tBdvsyIBpWTdU/edit?usp=sharing

## Model Performance using K-Fold Cross-Validation

| Classifier                        | 1-Fold Balanced Accuracy | 5-Fold Balanced Accuracy |
| --------------------------------- | ------------------------ | ------------------------ |
| Content Linear Softmax            | 80.99                    | 76.27                    |
| Neural Network with 3 dense layer | 84.87                    | 85.73                    |
| Neural Network with 5 dense layer | 85.57                    | 85.63                    |
| 1D-CNN with 2conv 1fc layer       | 87.11                    | 87.13                    |
| CNN with 5conv layer              | 83.29                    | 87.16                    |
| Random Forest                     | 80.69                    | 80.09                    |

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
