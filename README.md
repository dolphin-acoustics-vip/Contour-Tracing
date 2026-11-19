This repository houses ongoing workflows to automate time-frequency contour extraction for the analysis of narrow-band (tonal) bioacoustic signals, particularly delphinid whistles. Currently includes base files for whistle analysis using a combined python workflow developed by Dr Paul Best and Charlie Hargrave. Predict.py calls the crepe CNN model for frequency estimation. Details for its use, pretrained crepe weights and all prerequisites can be found at https://github.com/mim-team/bioacoustic_F0_estimation. Gui.py generates an interactive plotting window for the manual evaluation and editing of these model estimations to correct for errors and misplaced points (includes quality of life improvements made by Tamra Chan, incl. spectrogram customisation tools). ContourStats.py allows for the calculation of several time-frequency parameters used for tonal sound analysis in order to emulate those calculated by ROCCA for Pamguard, and is to be incorporated into the back end of the gui.py script.

***Students:*** *Please upload your own contributions and describe their content and use-cases in this README file, if using any external resources please cite or link them.*  

## Setup
Please make use of the requirements.txt to keep track of python libraries used in the repository. Setting up a python virtual environment is recommended, instructions on how to do so can be found at https://www.w3schools.com/python/python_virtualenv.asp. Name your venv "venv", otherwise update the gitignore so that you don't push your virtual environment to main. 

Once the venv is setup you can use `pip install -r requirements.txt` to install all packages required in the repository. If you add a new package use `pip freeze > requirements.txt` to update the required packages. 

