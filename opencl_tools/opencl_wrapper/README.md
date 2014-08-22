OpenCLWrapper
=============

Requirements
------------

OpenCLWrapper requires [boost][1] and OpenCL.
It relies on CMake. Out of tree compilation is strongly suggested.

Before compiling modify the main "CMakeLists.txt" adding the 
right paths to the path lists: OPENCL_INCLUDE_PATHS, OPENCL_LIB_PATHS
and BOOST_LIB_PATHS.

The "examples" directory contains the client code for 
matrix multiplication algorithm.

If you want to update the code maintain the same coding conventions.

Author
-------

For any questions contact [Alberto Magni][2]

[1]: http://www.boost.org/
[2]: mailto:alberto.magni86@gmail.com
