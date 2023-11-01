for attack in no_attack partial_krum full_krum partial_trim full_trim gaussian label_flipping alie_attack
do
    # filename="${dataset}_${model}_${algorithm}_${alpha}_${local_epoch}"
    ARGS="--gpu=1 --iid=0 --attack_type=${attack} --stdout=logs/ --dataset=udacity --model=dave2 --epochs=200"
    python byz_main.py ${ARGS}
done
