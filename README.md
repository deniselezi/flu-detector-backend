# flu-detector-backend

Backend for the flu detector web-app developed as part of my final year project at University College London (UCL).

This backend contains the models developed as part of the project, namely a Lasso model, a feed-forward neural network and a GRU. Each model is trained for ILI rate prediction with data from 2007 to 2017, and is deployed for testing on data from 2017 to 2019. 

`main.py` is the entry point fo the back-end and handles the routing. `modelling.py` handles the modelling. Models are stored in the `models/` directory.

To run the application, use `flask --app main run`.
