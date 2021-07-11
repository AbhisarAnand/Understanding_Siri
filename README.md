# Understanding Siri
### Thank you for visiting! ![](https://visitor-badge.glitch.me/badge?page_id=AbhisarAnand.Understanding_Siri) [![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)   
This repository contains code that takes in a text input and outputs the intent of the sentence and the supporting information.

## Running the Code

### Cloning the Code
```shell
~$ git clone https://github.com/AbhisarAnand/Understanding_Siri.git
```
### Initializing Git LFS
You can download and set up Git LFS from their [website](https://git-lfs.github.com/).
```shell
~$ git lfs install
```
Once you have initialized Git LFS, you will need to pull the weights of the model.
```shell
~$ git lfs pull
```

### Installing the packages
You can use ```pip3``` to install the packages required to run the package.
```shell
~$ pip3 install -r requirements.txt
```

### Running the program
```shell
~$ streamlit run main.py
```

### Deploying this program on the local server
Go to [ngrok.com/download](https://ngrok.com/download) and download the zip file for your OS.
Unzip the file
```shell
~$ unzip /path/to/ngrok.zip
```
Connect Your Account
```shell
~$ ./ngrok authtoken <your_auth_token>
```
Fire up the Server
```shell
~$ ./ngrok http <your_port_which_streamlit_is_using>
```
