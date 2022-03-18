#!/bin/bash

PYFILE='/home/robesafe/Miguel/sim-data-aug/data_split_txt.py'
DATASET=~/Miguel/datasets/vkitti/

tr_file=$DATASET/train.txt
te_file=$DATASET/test.txt
val_file=$DATASET/val.txt

touch tr_file
touch te_file
touch val_file

if [[ -s $te_file ]]
then
    cat /dev/null > $te_file
else
    touch $te_file
fi

if [[ -s $tr_file ]]
then
    cat /dev/null > $tr_file
else
    touch $tr_file
fi


if [[ -s $val_file ]]
then
    cat /dev/null > $val_file
else
    touch $val_file
fi

declare -a names=("15-deg-left" "15-deg-right" "30-deg-left" "30-deg-right" "clone" "morning" "overcast" "sunset")

cd $DATASET

for i in "${names[@]}"
do
    echo "Processing $i"
    python $PYFILE -dir "Scene01/$i/frames/rgb/" -tr 0.8 -val 0.1 -te 0.1
    cat "Scene01/$i/frames/rgb/test.txt" >> $DATASET/test.txt
    cat "Scene01/$i/frames/rgb/train.txt" >> $DATASET/train.txt
    cat "Scene01/$i/frames/rgb/val.txt" >> $DATASET/val.txt

    python $PYFILE -dir "Scene02/$i/frames/rgb/" -tr 0.8 -val 0.1 -te 0.1
    cat "Scene02/$i/frames/rgb/test.txt" >> $DATASET/test.txt
    cat "Scene02/$i/frames/rgb/train.txt" >> $DATASET/train.txt
    cat "Scene02/$i/frames/rgb/val.txt" >> $DATASET/val.txt

    python $PYFILE -dir "Scene06/$i/frames/rgb/" -tr 0.8 -val 0.1 -te 0.1
    cat "Scene06/$i/frames/rgb/test.txt" >> $DATASET/test.txt
    cat "Scene06/$i/frames/rgb/train.txt" >> $DATASET/train.txt
    cat "Scene06/$i/frames/rgb/val.txt" >> $DATASET/val.txt

    python $PYFILE -dir "Scene18/$i/frames/rgb/" -tr 0.8 -val 0.1 -te 0.1
    cat "Scene18/$i/frames/rgb/test.txt" >> $DATASET/test.txt
    cat "Scene18/$i/frames/rgb/train.txt" >> $DATASET/train.txt
    cat "Scene18/$i/frames/rgb/val.txt" >> $DATASET/val.txt

    python $PYFILE -dir "Scene20/$i/frames/rgb/" -tr 0.8 -val 0.1 -te 0.1
    cat "Scene20/$i/frames/rgb/test.txt" >> $DATASET/test.txt
    cat "Scene20/$i/frames/rgb/train.txt" >> $DATASET/train.txt
    cat "Scene20/$i/frames/rgb/val.txt" >> $DATASET/val.txt
done









