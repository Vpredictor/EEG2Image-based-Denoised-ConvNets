# Exploring-BCI-Control-in-Smart-Environment
EEG  Intention Recognition

### Overview
This project is going to classifiy 5-class EEG signals using  the EEG2Image based denoised-ConvNets model

### Environment
keras
tensorflow
sklearn
matplotlib
numpy

### Dataset

Two public EEG datasets involved in this paper are EEGMMIDB and EMOTIV.

### EEGMMIDB
  - The EEGMMIDB data set has a total of 29,738 pieces of data, each data is an EEG image with 32 pixels * 32 pixels * 3 channels.
  - These images can be divided into five categories. 0: Keep eyes closed; 1: Open and close both feet; 2: Open and close both fists; 3: Open and close left fist; 4: Open and close right fist.
  - The training data is stored in a Numpy-specific binary format npy file called data.npy, and the corresponding label is saved in another npy file called label.npy

 ### EMOTIV
  - The EEGMMIDB data set has a total of 5266 pieces of data, each data is an EEG image with 32 pixels * 32 pixels * 3 channels.
  - These images can be divided into five categories. 0: Confirm; 1: Up; 2: Down; 3: Left; 4: Right.
  - The training data is stored in a Numpy-specific binary format npy file called X_5000.npy, and the corresponding label is saved in another npy file called y_5000.npy

### Training 
Train EEGMMIDB: you need stn_train.py, eegmmidb_conv_model.py, spatial_transformer.py and run stn_train.py.
- Input: data.npy, label.npy
- Output: Training, Testing result and report

-Train EMOTIV: you need stn_train.py, emotiv_conv_model.py,  spatial_transformer.py and run stn_train.py.
- Input: X_5000.npy, y_5000.npy
- Output: Training, Testing result and report

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)


   [dill]: <https://github.com/joemccann/dillinger>
   [git-repo-url]: <https://github.com/joemccann/dillinger.git>
   [john gruber]: <http://daringfireball.net>
   [df1]: <http://daringfireball.net/projects/markdown/>
   [markdown-it]: <https://github.com/markdown-it/markdown-it>
   [Ace Editor]: <http://ace.ajax.org>
   [node.js]: <http://nodejs.org>
   [Twitter Bootstrap]: <http://twitter.github.com/bootstrap/>
   [jQuery]: <http://jquery.com>
   [@tjholowaychuk]: <http://twitter.com/tjholowaychuk>
   [express]: <http://expressjs.com>
   [AngularJS]: <http://angularjs.org>
   [Gulp]: <http://gulpjs.com>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>
