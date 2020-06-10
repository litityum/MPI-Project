# Given the following arguments;
# --- compiles $1 into $exe
# --- runs $exe on $2 processors, on input file $3 and for time-steps {$4,...,$5}
# --- NOTE: it deliberately recalculates from scratch
# --- prints the output into images basename($3)_{$4,...,$5}.png, in the folder $6
# $1: source file
# $2: number of processors
# $3: input file
# $4: the time-step to start with
# $5: the time-step to end at
# $6: where to output
# $7: scaling factor (optional, default=4)

sf=${7:-4}

# compiles the code
exe="$( basename $1 .c )"
mpicc $1 -lm -o $exe

# gets the basename of $3
imgprefix="$( basename $3 | sed 's/\..*//' )"

# Thanks to: https://stackoverflow.com/a/8789815/2736228
# iterating from $4 to $5, all padded with 0s to width 3
for i in $(seq -f "%03g" $4 $5)
do
	mpirun -np $2 --oversubscribe ./$exe $3 /dev/stdout $i | # output file as /dev/stdout, piped to...
		sh convert.sh /dev/stdin "$6/${imgprefix}_$i.png" $sf # see conv.sh for its details
done
