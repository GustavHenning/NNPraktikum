#!/bin/bash

QUIET=False
TIME=False

function run {
	if [[ $QUIET == True ]]; then
			run="$1: ""$(python Run.py $1 -ns | grep % | cut -d':' -f 2 | sed 's/^ *//;s/ *$//')"
	else
			echo "$1: ""$(python Run.py $1 -ns)"
		fi

			echo $run
}


while test $# -gt 0
do
    case "$1" in
        -q) QUIET=True
            ;;
        -t) TIME=True
            ;;
        --*) echo "bad option $1"
            ;;
        *) echo "argument $1"
            ;;
    esac
    shift
done

#function runAll	{
#}
	if [[ $TIME == True ]]; then
		time {
			run "2,1"
			run "2,2,1"
			run "10,1"
			run "5,4,3,2,1"
		}
	else
		run "2,1"
		run "2,2,1"
		run "10,1"
		run "5,4,3,2,1"
	fi
