steps=(5 10 25 50 100 200)
models=("EDM-L128" "EDM-L128" "EDM" "VE" "VP")

for model in "${models[@]}"
 do
   for step in "${steps[@]}"
   do
    python /path/to/test_segmentation.py \
    --config=/path/to/SEGMENTATION_CFG.py \
    --data-dir=/path/to/data/folder/"${model}"/"${model}"_"${step}" \
    --load-path=/path/to/model/save/folder/"${model}"_"${step}"/model.pth
    done
  done
