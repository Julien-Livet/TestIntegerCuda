#!/bin/bash
#nvcc --expt-relaxed-constexpr -rdc=true --std c++20 -v -Xcompiler -Wall -w -Xcompiler -v --ptxas-options=-v main.cu -o TestIntegerCuda
#nvcc --expt-relaxed-constexpr -rdc=true --std c++20 -Xcompiler -Wall -w main.cu -o TestIntegerCuda
nvcc --expt-relaxed-constexpr -rdc=true --std c++20 -Xcompiler -Wall main.cu -o TestIntegerCuda
