OMP_NUM_THREADS=10 torchrun --standalone --nproc_per_node=1  /path/to/train_autoencoder.py \
--config=/path/to/VAE_64_CFG.py \
--data-dir=/path/to/CAMUS \

