##########################
## PreProcess
##########################

# Sample draw the fonts and save to paired_images, about 10-20 mins
# /usr/local/opt/python@3.7/bin/python3.7 font2img_kai.py


##########################
## Train and Infer
##########################

for i in {1..4};
do
    # Train the model
    /usr/local/opt/python@3.7/bin/python3.7 train.py --experiment_dir=experiments \
                    --experiment_id=1 \
                    --batch_size=64 \
                    --lr=0.001 \
                    --epoch=1 \
                    --sample_steps=50 \
                    --schedule=20 \
                    --L1_penalty=100 \
                    --Lconst_penalty=15 | tee -a shlog.log

    # Infer
    /usr/local/opt/python@3.7/bin/python3.7 infer.py --model_dir=experiments/checkpoint/experiment_1 \
                    --batch_size=32 \
                    --source_obj=experiments/data/val.obj \
                    --embedding_ids=136 \
                    --save_dir=save_dir/

    /usr/local/opt/python@3.7/bin/python3.7 infer_by_text.py --model_dir=experiments/checkpoint/experiment_1 \
                    --batch_size=32 \
                    --embedding_id=136 \
                    --save_dir=save_dir/

    echo "One More again"
    echo ${i}
    echo ""
    echo ""

    # find ./save_dir -name '*.png' -type f -not -path "*/step*" \
    #     -exec sh -c 'mv {} ./save_dir/$(echo ${i})_$(basename -s .png {}).png' \;

    FILES=$(find ./save_dir -name 'inferred*.png' -type f -not -path "*/step*")
    for f in $FILES
    do
        name=$i\_$(basename -s .png $f)
        echo "mv $f -> $name"
        mv "$f" "./save_dir/${name}.png"
    done
done

##########################
## Finetune
##########################

# Generate paired images for finetune
# /usr/local/opt/python@3.7/bin/python3.7 font2img_finetune.py


# Train/Finetune the model
# /usr/local/opt/python@3.7/bin/python3.7 train.py --experiment_dir=experiments_finetune \
#                 --experiment_id=0 \
#                 --batch_size=16 \
#                 --lr=0.001 \
#                 --epoch=1 \
#                 --sample_steps=1 \
#                 --schedule=20 \
#                 --L1_penalty=100 \
#                 --Lconst_penalty=15 \
#                 --freeze_encoder_decoder=1 \
#                 --optimizer=sgd \
#                 --fine_tune=67 \
#                 --flip_labels=1

# /usr/local/opt/python@3.7/bin/python3.7 infer.py --model_dir=experiments_finetune/checkpoint/experiment_0 \
#                 --batch_size=32 \
#                 --source_obj=experiments_finetune/data/val.obj \
#                 --embedding_id=67 \
#                 --save_dir=save_dir/



