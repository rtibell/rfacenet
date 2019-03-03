#!/bin/bash

for i in `grep "PASS" /home/pi/Projects/Movidius/RTstuff/facenet/${1} | awk '{printf("%s %s\n",$3,$5);}' | sed -e 's/_a././' | sed -e 's/_/ /g' | awk '{printf("%s_%s\n",$1,$4);}'`
do
    p1=`echo $i | sed -e 's/_/ /' | awk '{printf("%s\n",$1);}'`
    p2=`echo $i | sed -e 's/_/ /' | awk '{printf("%s\n",$2);}'`
    if [ $p1 =  $p2 ]; then
        echo "EQ $p1  $p2"
    else
        echo "NEQ $p1  $p2"
    fi
done
