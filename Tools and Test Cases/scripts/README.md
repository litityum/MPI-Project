## Prerequisites

To use the scripts `convert.sh` and `sequence.sh`,
you will first need to install a package called `netpbm`.

### Installing `netpbm` on Linux

Use your package manager. On WSL or the VM provided,
the following command does it:

```
sudo apt install netpbm
```

### Installing `netpbm` on macOS

Obtain Homebrew from the following link: https://brew.sh/

Then, use the following command:

```
brew install netpbm
```

## How to use the scripts

For each script,
there is a more detailed explanation on the arguments
as comments in the scripts.

### Converting a map.txt into map.png via `convert.sh`

The last argument is optional, and by default is `4`.
The syntax is:

```
sh convert.sh [inputfile] [outputfile] [scalingfactor]
```

Scaling factor is how many pixels a cell will take in the image,
both in width and height.
Following is an example use of command:

```
sh convert.sh map.txt map.png 6
```

### Compilation, repeated execution, and repeated conversion via `sequence.sh`

Note that this script works with the C source codes only.
You may modify it easily to work with C++ and Python.

The last argument is optional, and by default is `4`.
The syntax is:

```
sh sequence.sh [sourcefile] [numberofprocessors] [inputfile] [t_start] [t_end] [output_directory]
```

The example use is as follows:

```
sh sequence.sh game.c 5 rand.txt 10 25 ~/some/directory
```

This script does the following:

1. Compiles the `game.c` into `game`.
2. Using `game`, simulates `rand.txt` for 10, 11, 12, ..., 25 time-steps.
Each simulation starts from the beginning, deliberately inefficient for testing purposes.
3. Prints the outputs of simulations via `convert.sh` as `rand_010.png`, ..., `rand_025.png`.
Files are placed into `~/some/directory`.

