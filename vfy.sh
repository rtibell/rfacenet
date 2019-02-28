#!/bin/bash

echo "Intersect"
sh vfy-item.sh intersect ${1} | awk '{printf("%s\n",$1);}' | sort | uniq -c

echo "Near"
sh vfy-item.sh near ${1} | awk '{printf("%s\n",$1);}' | sort | uniq -c

echo "Org"
sh vfy-item-org.sh ${1} | awk '{printf("%s\n",$1);}' | sort | uniq -c

