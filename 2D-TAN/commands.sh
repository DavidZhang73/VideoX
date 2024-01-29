CUDA_VISIBLE_DEVICES=2 && PYTHONPATH=/home/users/u6921098/project/VideoX/2D-TAN/lib:$PYTHONPAT && python moment_localization/train.py --cfg experiments/tacos/2D-TAN-128x128-K5L8-pool.yaml --verbose

CUDA_VISIBLE_DEVICES=3 && PYTHONPATH=/home/users/u6921098/project/VideoX/2D-TAN/lib:$PYTHONPAT && python moment_localization/train.py --cfg experiments/tacos/2D-TAN-128x128-K5L8-conv.yaml --verbose
