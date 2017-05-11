#!/bin/bash

#
# Installs the prerequisites for this project on an Ubuntu/Debian environment
# author: Gustav Henning
#

function pkgExists (){
	local res
	res=$(dpkg -l $1 | grep $1 | wc -l)
	echo $res #returns >=1 if package is installed, else 0
}

dpkgExists=$(which dpkg | wc -l)  # package manager exists

if [ "$dpkgExists" -ne "0" ]; then
	# python
	hasPy=$(pkgExists "python")
	if [ "$hasPy" -eq "0" ]; then
		echo "Python cannot be found, install it and try again"
		exit 1
	else
		echo "Python installation found..."
	fi
	# pip
	hasPip=$(which pip | wc -l)
	if [ "$hasPip" -eq "0" ]; then
		echo "You need to install pip to continue"
		echo "We'll try automatically, if you'd like to cancel, Ctrl+C now"
		read -n 1 -s -p "Press any key to continue..."
		python get-pip.py
	else
		echo "Pip installation found..."
	fi
	echo "Upgrading pip..."
	python -m pip install --upgrade pip
	echo "Installing required pip packages for numpy etc..."
	pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose
	# Since we installed on the user permissions we have to make sure we have
	# the correct $PATH set up. Source: https://scipy.org/install.html
	hasPathCfg=$(echo $PATH | grep "$USER/.local" | wc -l)
	if [ "$hasPathCfg" -eq "0" ]; then
		echo "Because pip installed on user side, we need additional PATH vars."
		echo "We'll try automatically(export & ~/.bashrc)"
		echo "if you'd like to cancel, or you know you have done this before, Ctrl+C now"
		read -n 1 -s -p "Press any key to continue..."
		export PATH="$PATH:/home/$USER/.local/bin"
		echo "export PATH=\"$PATH:/home/$USER/.local/bin\"" >> ~/.bashrc
	else
		echo "Path config found..."
	fi
	# scikit-learn
	echo "Installing scikit-learn..."
	# Further reading: http://scikit-learn.org/stable/install.html
	pip install --user scikit-learn
	if [ "$?" -eq "0" ]; then
			echo "Installation complete"
	else
			echo "scikit-learn installation failed, something went wrong."
	fi
fi
