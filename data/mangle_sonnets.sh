cat sonnets.txt | tr '[:lower:]' '[:upper:]' | tr -d '[:punct:][:digit:]' | tr ' ' '\n' | tr -s '\n'  > sonnets-uppercase.txt
