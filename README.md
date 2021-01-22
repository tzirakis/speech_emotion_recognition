# End-to-end Speech Emotion Recognition using Deep Neural Networks 

This repository provides training and evaluation code for our end-to-end speech emotion recognition paper. If you use this codebase in your experiments please cite:

`Tzirakis, P., Zhang, J. and Schuller, B.W., 2018, April. End-to-end speech emotion recognition using deep neural networks. 
In 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 5089-5093).`

(https://ieeexplore.ieee.org/abstract/document/8462677)

### Implementation of this method in PyTorch (along with pretrain models) can be found in our [End2You toolkit](https://github.com/end2you/end2you)

## Requirements
Below are listed the required modules to run the code.

  * Python <= 2.7
  * NumPy >= 1.11.1
  * TensorFlow <= 0.12
  * MoviePy >= 0.2.2.11
 
## Content
This repository contains the files:
  * model.py: Contains the audio network.
  * emotion_train.py: Performs the training of the model.
  * emotion_eval.py: Performs the evaluation of the model.
  * data_provider.py: Provides the data.
  * data_generator.py: Creates the tfrecords from '.wav' files
  * metrics.py: Contains the CCC metric that is used during evaluation.
  * losses.py: Contains the CCC loss function that is used during training.
