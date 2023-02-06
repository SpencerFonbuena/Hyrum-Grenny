# OM-Manufacturing Repo contains three examples of my most recent efforts to form a more data centered trading strategy. 

full_analysis.py

This is my first python project trying to test the predictive validity of RSI divergence on Index futures. It is very sloppy, but it represents where I started. 

createdataset.py

This file was used to create a dataset on a VM to feed into a convolutional neural network to check the statistical validity of oculular analysis when trading index futures.

trainnetwork.py

This is an application of a slightly modified VGG16 architecture that uses the data from the "createdataset.py" dataset and tries to predict price movement with computer vision.
