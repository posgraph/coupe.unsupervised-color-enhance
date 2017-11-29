CUDA_VISIBLE_DEVICES=0 python main.py --dataset_dir=color_enhance --phase=test --model_dir=color_enhance_512 \
--lr=0.0001 --dataset_root=./datasets --batch_size=1 --load_size=360 --fine_size=300 --print_freq=100
