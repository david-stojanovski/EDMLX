steps=(5 10 25 50 100 200)
models=("EDM-L128" "EDM-L128" "EDM" "VE" "VP")

for model in "${models[@]}"
 do
   for step in "${steps[@]}"
   do
    python /path/to/train_classification.py \
    --config=/path/to/CLASSIFICATION_CFG.py \
    --data-dir=/path/to/classification_folder/"${model}"/"${model}"_"${step}" \
    --results-dir=/path/to/results/folder/"${model}"_"${step}"
    done
  done