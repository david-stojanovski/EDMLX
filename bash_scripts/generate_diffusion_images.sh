steps=(5 10 25 50 100 200)
models=("EDM-L128" "EDM-L128" "EDM")

for model in "${models[@]}"
 do
   for step in "${steps[@]}"
   do
    OMP_NUM_THREADS=10 torchrun --standalone --nproc_per_node=6 /path/to/generatev2.py \
    --data-dir=/path/to/inf_data_folder \
    --steps="${step}" \
    --load-diffusion-model-pth=/path/to/"${model}"/best_val_iter.pkl \
    --results-dir=/path/to/results/folder/"${model}"_"${step}"
    done
  done