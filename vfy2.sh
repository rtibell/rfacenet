#!/bin/bash

echo "Query"
sh vfy-item2.sh match ${1} | awk '{printf("%s\n",$1);}' | sort | uniq -c


