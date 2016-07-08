#!/bin/bash -i
DATA_PATH='../results/'

read -e -p "Enter which program you want to run: " -i filename
read -e -p "Enter a filename for the log: " -i log
touch $DATAPATH/$log

run() {
    echo -e "Running " + $filename + "...\n"
    if python2.7 $filename >> $DATA_PATH/$log ; then
        echo -e "Done\n"
    else
        read -n1 -rsp "Execution failed... Fix your errors and press [RET]" key
        if [[ "$key" == "" ]]; then 
            run
        else
            echo "Exiting Program"
            rm $DATA_PATH/$filename
}
