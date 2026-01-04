# Pi Calculator

This repository contains programs calculating Pi to arbitrary precisions based on
the Chudnovsky Algorithm. The general program structure was copied from the
[Wikipedia article][wikipage]. This is mainly intended as an exercise for me to
learn about tools and languages.

## Python 

mpmath is required to run this program. The current version does not work due to
the binary split function exceeding Python's maximum recursion depth.

## C

The C program is written on a Linux system. A Makefile is supplied to compile
it, and the only external dependency is GNU GMP. A simple script (benchmark.sh)
is also included to test runtime vs precision.


[wikipage]: https://en.wikipedia.org/w/index.php?title=Chudnovsky_algorithm&oldid=1245941440
