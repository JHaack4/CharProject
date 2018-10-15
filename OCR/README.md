This guide assumes that the user  is using an ubuntu 16.04 computer. If the user is using a linux operating system, this guide should still be almost correct. Personal experience recommends using an LTS version of ubuntu and not the newest possible one. Other operating systems may be more complicated. We will also assume the user is using python 3. The Nvidia drivers, Cuda, and CudNN are only necessary if the user wishes to use GPUs. When using Nvidia graphics cards with ubuntu 16.04 you may need to set a certain boot option to prevent the computer from crashing. The boot option is nomodeset. You can set it to run by default by going to /etc/default/grub and modifying the line GRUB_CMDLINE_LINUX_DEFAULT by adding nomodeset as one of the options. The first time you boot the computer when installing ubuntu, if you already have the nvidia graphics placed, you may need to set the boot option for the first time you start the OS (otherwise you may crash just trying to install it). You should also turn off secure boot from either uefi/bios depending on which one the computer has.

Dependencies:
	1. Nvidia drivers
	2. Cuda
	3. CudNN
	4. Pytorch
	5. OpenCV
	6. Pillow
	7. WarpCTC
	8. LMDB
	9. numpy
	10. scipy

Instructions:
	1. Nvidia drivers
		For most gpus you can find the most recent nvidia drivers at http://www.nvidia.com/Download/index.aspx. Check that page and try to find your gpu and make sure you pick the drivers for the right operating system. Once you download the drivers from that site, you can just run the file it gives you from the terminal and it should then install the drivers for you.

		After you install the drivers you should go to additional drivers and make sure that the new driver is being used. You may need to restart after this is done for it to take effect. You may run into issues involving nouveau drivers and I'd recommend just looking up how to disable them if needed. You may also need to stop lightdm to get this to work. One page that talks about the install process in a lot more detail is here:

		https://gist.github.com/wangruohui/df039f0dc434d6486f5d4d098aa52d07


	2. Cuda
		The most recent version of cuda that works with pytorch at the moment of writing this is 9.1. You can see what versions of cuda pytorch supports
        by going to this page https://pytorch.org/. The link for installing cuda 9.1 is this, https://developer.nvidia.com/cuda-downloads.

		Choose the relevant options. For the installer type I'd recommend either the runfile local or deb local. There is a more
		detailed guide here,

		https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

		There are some instructions in the post installation section that need to be followed. Specifically, the path and
		LD_LIBRARY_PATH environment variables need to be modified in either the .bashrc or .bash_profile or .profile file (if you aren't using bash as your shell find the relevant equivalent file that is run when you start the shell). As for which file to modify, this page has a discussion on the differences (the recommendation at the bottom is the most relevant part):

		http://www.joshstaiger.org/archives/2005/07/bash_profile_vs.html

		The lines that need to be added assuming you installed in the default location are:

		export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}

		For 64 bit OS:

		export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

		For 32 bit OS:

		export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

		From personal experience you may have issues installing cuda due to nouveau display drivers that ubuntu default comes with. Check the second url (the long nvidia guide)
		for information about how to deal with that issue if it occurs. There are also some pre-installation instructions you should glance at, but typically they are satisfied assuming you installed the nvidia drivers first.


	3. CudNN
		This page has the installation instructions for cudnn:

		http://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installlinux

		Any version works, so I recommend just using the newest that is compatabile with your version of cuda.


	4. Pytorch
		The current state of the code base means that using the most recent version of pytorch will lead to at least deprecation warnings (mainly from Variable usage). The most recent version of pytorch that the code base has been tested with is pytorch 0.3.1. Instructions to install this version can be found on this base. https://pytorch.org/previous-versions/

	5. OpenCV
		- pip3 install opencv-python

	6. Pillow
		- pip3 install Pillow

	7. WarpCTC
		This guide is based off of the one found here https://github.com/SeanNaren/warp-ctc/tree/pytorch_bindings/pytorch_binding
		Commands:
			git clone https://github.com/SeanNaren/warp-ctc.git
			cd warp-ctc
			mkdir build; cd build
			cmake ..
			make
		If using Cuda make sure CUDA_HOME is set correctly. The most common way of setting it, if you did the default way of installing cuda would be:
			export CUDA_HOME="/usr/local/cuda"
		Lastly, to install the pytorch bindings do the following:
			cd pytorch_binding
			python3 setup.py install

	8. LMDB
		- pip3 install lmdb

	9. numpy
		- pip3 install numpy

	10. scipy
		- pip3 install scipy
