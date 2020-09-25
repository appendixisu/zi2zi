for i in {1..4};
do
    FILES=$(find ./save_dir -name 'inferred*.png' -type f -not -path "*/step*")
    for f in $FILES
    do
        name=$i\_$(basename -s .png $f)
        echo "mv $f -> $name"
        mv "$f" "./save_dir/${name}.png"
    done
done