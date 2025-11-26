cat shakespeare-complete.txt| sed s/-/\ /g | tr '[:lower:]' '[:upper:]' | tr -d '[:punct:][:digit:]' | tr ' ' '\n' | sort | uniq > shakespeare-complete-unique-uppercase.txt 

