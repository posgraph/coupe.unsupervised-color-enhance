CUDA_VISIBLE_DEVICES=0 python main.py --dataset_dir=color_enhance --model_dir=color_enhance_512 \
--lr=0.0001 --dataset_root=/data1 --batch_size=2 --load_size=512 --fine_size=256 --print_freq=100
