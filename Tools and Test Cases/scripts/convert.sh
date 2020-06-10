# Thanks to: https://stackoverflow.com/a/4872225/2736228
# requires netpbm
# $1: input file
# $2: output file
# $3: scaling factor (optional, default=4)

# Thanks to: https://stackoverflow.com/a/9333006/2736228
# Scaling factor
sf=${3:-4}

# printf produces empty space of length $sf
# sed replaces each space character with '\1' or 'p;'
# RHS produces "\1\1\1..." and "p;p;p;..." with $sf repetitions
colrepeat="$( printf '%*s' "$sf" | sed 's/ /\\1/g' )"
rowrepeat="$( printf '%*s' "$(( sf-1 ))" | sed 's/ /p;/g' )"

# Thanks to: https://serverfault.com/a/915814
# awk 'NF' removes the empty lines
map="$( awk 'NF' $1 )"

# map and image sizes
S="$( echo "$map" | wc -l )"
IS=$(( S * sf ))

echo "$map" |
	sed -e 's/ //g' | # removes the spaces
	# scaling:
	sed -e "s/\([01]\)/$colrepeat/g" | # repeats each 0 or 1 in row $sf times
	sed -e "$rowrepeat" | # repeats each line $sf times
	# adherence to pnm file format syntax:
	sed -e "1i\\$IS $IS" | # inserts "(S*sf) (S*sf)" on top of the file
	sed -e '1i\P1' | # inserts "P1" on top of the file
	# conversion to png
	pnmtopng > $2
