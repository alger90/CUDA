
############ ABOUT HOW TO RUN THE PROGRAM #################
1. Our program requires the pre-installation of libpng. If you are using linux, please use apt-get to install it.

2. When libpng is installed, change the make file so as to indicate the location of the libpng library location correctly on your machine. Please go to Program Code/<Program You Choose>/Release/, and open the makefile, changing the "-L/opt/local/lib -lpng15" to right location. 

3. If your machine is 32 bit machine, you have to change the -m64 to the right flag too. 

4. The header files of libpng should also be added into the include path correctly. One simple way is to open the subdir.mk and change "-I/opt/local/include -I/opt/local/include/libpng15 " to right value of both commands. Don't forget to change the -m64 if necessary.


############# About the Team organization ###################
The team work is planned to be split into equal parts. However, it turned out not every one is active on the project. We simply list our team members and their contribution as follows,

Hang Gao 80%
Lianjie Sun 10%
Tsu Chen 10%