#!/bin/bash -i
DATA_PATH='../results'

run_script() {
    echo -e "Running "  $filename
    if python2.7 $filename >> $DATA_PATH/$log ; then
        echo -e "Done\n"
    else
        read -n1 -rsp "Execution failed... Fix your errors and press [RET]" key
        if [[ "$key" == "" ]] ; then 
            run_script
        else
            echo "Exiting Program"
            rm $DATA_PATH/$log
        fi
    fi
}

read -e -p "Enter which program you want to run: " -i '' filename
read -e -p "Enter a filename for the log: " -i '' log
touch $DATA_PATH/$log
run_script

