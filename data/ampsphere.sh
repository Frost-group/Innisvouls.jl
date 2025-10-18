wget https://ampsphere-api.big-data-biology.org/v1/downloads/AMP.tsv

cat AMP.tsv | awk '{print $2}' | grep -v sequence  > ampsphere.dat

#histogram of length
awk ' {print(length($0))} ' ampsphere.dat | sort -n | uniq -c

