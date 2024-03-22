input_file=$1
per_file_len=$2
index_file=$3

## demo
##input_file="./dict/tuwen_info.txt"
##index_file="dict/file_indices"
##per_file_len=20000


file_len=`wc -l $input_file| cut -d " " -f 1`
let x=$file_len/$per_file_len
echo $x
seq=`seq 0 $x`

split -l $per_file_len -d -a3 $input_file $input_file

rm $index_file

for i in ${seq[@]}
do
    if [[ $i -lt 10 ]]; then
        idx=00${i}
    
    elif [[ $i -lt 100 ]]; then
        idx=0${i}

    else
        idx=${i}
    fi
    echo $idx >> $index_file
done

