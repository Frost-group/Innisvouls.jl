#wget https://ampsphere-api.big-data-biology.org/v1/downloads/AMP.tsv

cat AMP.tsv | awk '{print $2}' > ampsphere.dat

