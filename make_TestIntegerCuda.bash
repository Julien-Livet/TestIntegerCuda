#!/bin/bash
#Release
#nvcc -D__CDPRT_SUPPRESS_SYNC_DEPRECATION_WARNING -DCUDA_FORCE_CDP1_IF_SUPPORTED -diag-suppress 23,63,2361 --expt-relaxed-constexpr -rdc=true --std c++20 -DNDEBUG -Xcompiler -Wall main.cu -o TestIntegerCuda
#Debug
nvcc -D__CDPRT_SUPPRESS_SYNC_DEPRECATION_WARNING -DCUDA_FORCE_CDP1_IF_SUPPORTED -G -g -diag-suppress 23,63,2361 --expt-relaxed-constexpr -rdc=true --std c++20 -Xcompiler -Wall main.cu -o TestIntegerCuda
