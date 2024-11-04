# Intro to CUDA C and GPU Architecture Course - 6 hr and extended versions 

Note: Book used for this course is "Programming Massively Parallel Processors - A Hands-on Approach" 
The current newest edition is the 4th edition. However, there is a free PDF of for the 4th edition. So, feel free
to use the free PDF found here:

http://gpu.di.unimi.it/books/PMPP-3rd-Edition.pdf

## Links to 3rd edition and materials:

https://shop.elsevier.com/books/programming-massively-parallel-processors/kirk/978-0-12-811986-0

## Book resources root links:
https://booksite.elsevier.com/9780128119860/  
https://booksite.elsevier.com/9780128119860/lecture.php   (Extra Lecture Slides)

## Labs for Course link:
https://github.com/R100001/Programming-Massively-Parallel-Processors/tree/master

## CUDA C++ Programming Guide link:
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html



# Course Outline:
- nVidia GPU Architecture to Support CUDA
- Intro to CUDA C and Host program format
- CUDA threads, blocks, and indexing
  - nvcc compiler
  - kernel launch
  - memory management
  - kernel and host code synchronization
- Tensor Cores: Architecture and NN Application
- CUDA Memory Hierarchy
- Shared memory and thread synchronization
DRAM Circuit Operation Considerations: Access Types, Latency Caused by Non-Batch Fetches 
- Performance Considerations
- Brief Coverage of PyTorch with CUDA, cuDNN, and cuVSLAM

The course uses Jupyter Notebook - Colab. If you have your a GPU on your computer and wish to use a different application or command line execution, please feel free to do so.

* Follow these directions to get acquanted with running CUDA code on the Jupyter Notebook platform: \
https://www.geeksforgeeks.org/how-to-run-cuda-c-c-on-jupyter-notebook-in-google-colaboratory/


## Key Course Takeaways
- Why Nvidia GPU Architectures Changed to General Purpose Processing Architectures (CUDA Arch) 
- CPU vs GPU Hardware Architecture: Key differences in unit processor's complexity and why.
- GPU Hardware Basic Components Used For CUDA General Purpose Processing
- GPU Hardware to Software Vocabular Mapping/Translation
- Block and Thread Level Indexing Concept (through Lecture and Programming Homework Problems).
- Memory Hierarchy
- DRAM Circuit Operation Considerations: Access Types, Latency Caused by Non-Batch Fetches  
- Memory Coalescing vs Non-Coalesced Access Pattern Impact on Performance
- Performance Considerations
- CUDA, Numba, Cupy, Tensorflow, Pytorch relations

## Prerequisites
- Working Knowledge of C
- Exposure to Basic Computer Architecture

## GPU Access for Gaining Programming Experience 
Methods:
1) Nvida GPU installed on your own computer (via Windows, Linux, or Mac OS)
2) Use of Google Colab-Notebook through your web browser to gain free access of GPU via Cloud Service.

## Installing CUDA on Windows
https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html

## Verify CUDA Install (RHEL or Ubuntu) and Toolkit
https://xcat-docs.readthedocs.io/en/stable/advanced/gpu/nvidia/verify_cuda_install.html

## Other Links of Possible Interest or Reference
https://pytorch.org/docs/stable/notes/cuda.html



# Signup Instructions
Information about signing up for the 5hr courses I offer (The Zeno Institute of Robotics and Artificial Intelligence), is as follows:

You can purchase some of these courses through the [ZenoRobotcs.com](https://www.zenorobotics.com/courses) website. Other payment option is through Venmo, PayPal, Zelle, or Cash app. There is a $5/course savings for using Zelle. Please contact me for payment details for non-website methods.

Once you pay, I will send you a link to the booking calendar where you can setup your times.

When selecting your hour slots, please only choose an hour block for the first hour meeting. This will give me a chance to find out about your HW & SW setup, point out links to get you started, etc. Please limit any single day session to 2 hours max to give you time to absorb the concepts and do some programming/homework problems. Also, you don’t have to book all 5 hr time slots at once. You can select them as time progresses if you wish.



# Additional Learning Resources Links

## YouTube
Tom Nurkkala - Video talks for various Computer Science courses at Taylor University:

- CUDA Hardware \
  https://www.youtube.com/watch?v=kUqkOAU84bA

- Intro to GPU Programming \
  https://www.youtube.com/watch?v=G-EimI4q-TQ

CUDA University Courses

University of Illinois : Current Course: ECE408/CS483
Taught by Professor Wen-mei W. Hwu and David Kirk, NVIDIA CUDA Scientist. \
https://developer.nvidia.com/educators/existing-courses#2

Other:

- Data Access Pattern Matters: How CUDA Programming Works | GTC 2022 (6:55 and on) \
  https://www.youtube.com/watch?v=n6M8R8-PlnE

- Tutorial: CUDA programming in Python with numba and cupy: \
  https://www.youtube.com/watch?v=9bBsvpg-Xlk


## Code Links

- CUDA Samples \
  https://github.com/nvidia/cuda-samples
 
- Programming-Massively-Parallel-Processors Learning Material (Reading/Images, Exercises, & Labs) \
  https://github.com/R100001/Programming-Massively-Parallel-Processors/tree/master

- CUDA Concepts Cheat Sheet \
  https://kdm.icm.edu.pl/Tutorials/GPU-intro/introduction.en/
 
## Colab

- How to Use a GPU In Google Colab \
  https://www.geeksforgeeks.org/how-to-use-gpu-in-google-colab/  \
  https://www.geeksforgeeks.org/how-to-run-cuda-c-c-on-jupyter-notebook-in-google-colaboratory/

- How to Use Colab  \
  https://www.geeksforgeeks.org/how-to-use-google-colab/

- How to use GPU acceleration in PyTorch \
  https://www.geeksforgeeks.org/how-to-use-gpu-acceleration-in-pytorch/
 
- Colab Site \
  https://colab.research.google.com
 
- Example CUDA GPU Use Github/Notebook \
  https://colab.research.google.com/github/ShimaaElabd/CUDA-GPU-Contrast-Enhancement/blob/master/CUDA_GPU.ipynb#scrollTo=mgH5HreZ2WS9

- Example: GPU calculation in python with Cupy and Numba \
  https://colab.research.google.com/drive/15IDLiUMRJbKqZUZPccyigudINCD5uZ71?usp=sharing


## PTX and SASS

- Parallel Thread Execution (PTX)   \
  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
  
- PTX and SASS Assembly Debugging \
  https://docs.nvidia.com/gameworks/content/developertools/desktop/ptx_sass_assembly_debugging.htm


## PyCUDA

https://pypi.org/project/pycuda/


## Cupy

- About \
https://cupy.dev/

- Interoperability \
  https://docs.cupy.dev/en/stable/user_guide/interoperability.html






