# Repo with a few examples of personal quantitative research. Other Larger Projects are referenced below. 

full_analysis.py

This is my first python project trying to test the predictive validity of RSI divergence on Index futures. It is very sloppy, but it represents where I started. 

createdataset.py

This file was used to create a dataset on a VM to feed into a convolutional neural network to check the statistical validity of oculular analysis when trading index futures.

trainnetwork.py

This is an application of a slightly modified VGG16 architecture that uses the data from the "createdataset.py" dataset and tries to predict price movement with computer vision.

https://github.com/SpencerFonbuena/CNN-Softmax

CNN-Softmax classifier doing similar things as the two previously mentioned packages

https://github.com/SpencerFonbuena/DNNM

Nerual network based off the transformer. The idea is to use temporal and spacial information from multivariate market data to try and predict the next 'state' of the stock market. Improved upon previous paper by adding in convolutional layers at the end to prevent gradient death through softmax bottleneck

https://github.com/SpencerFonbuena/CFC

Another Neural Network based off of encoder only transformers to classify the state of the stock market. 

https://github.com/SpencerFonbuena/Jupyter-Insights

Jupyter notebooks of my own writeups of common topics in DNN research.

I have more projects available for discussion, but that are kept private because they are ongoing. 
