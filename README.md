In order to get the notebooks running on your machine you need to complete three steps:

1. clone the repository to your machine 
2. install python and dependencies
3. run the jupyter notebook command from a terminal

These steps are explained below.

-----

1. Cloning the repository from github

In order to clone this repository onto your own machine, you will need to install git. You can do so by following the instructions on this website:

https://linode.com/docs/development/version-control/how-to-install-git-on-linux-mac-and-windows/

After you have installed git, open a terminal and navigate to the folder you wish to clone the repository into. Then type the command:

git clone https://github.com/cambridge-mlg/online_textbook.git

The online_textbook folder should now be cloned into the directory you were in. The notebooks are about 200MB.

-----

2. Installing Python and dependencies

In order to run the notebooks, you will need to install python 3 if you haven't done so already. You can download python from this link:

https://www.python.org/downloads/

After you have installed python, you will need to install a few dependencies. To install these, or check that they are already installed, type the following command into your terminal:

python3 -m pip install scipy numpy matplotlib pandas

This should return a "Succesfully installed" message. 

-----

3. Running and accessing the notebooks

Finally, in order to open the notebook, you need to run, in the terminal, the command:

jupyter notebook

which will open a directory tree in jupyter notebooks in your browser. From here, navigate to the mphil-intro-module folder and open the file 'index.ipynb'. From here you should be able to navigate to all of the other notebooks. The home button in each notebook will take you back to this index page.


