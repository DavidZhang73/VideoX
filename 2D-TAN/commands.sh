PYTHONPATH=/home/users/u6921098/project/VideoX/2D-TAN/lib; python moment_localization/train.py --cfg experiments/tacos/2D-TAN-128x128-K5L8-pool.yaml --verbose

PYTHONPATH=/home/users/u6921098/project/VideoX/2D-TAN/lib; python moment_localization/train.py --cfg experiments/tacos/2D-TAN-128x128-K5L8-conv.yaml --verbose

PYTHONPATH=/home/users/u6921098/project/VideoX/2D-TAN/lib; python moment_localization/train.py --cfg experiments/iaw/2D-TAN-pool.yaml --verbose

PYTHONPATH=/home/users/u6921098/project/VideoX/2D-TAN/lib; python moment_localization/train.py --cfg experiments/iaw/2D-TAN-conv.yaml --verbose

PYTHONPATH=/home/users/u6921098/project/VideoX/2D-TAN/lib; python moment_localization/test.py --cfg experiments/iaw/2D-TAN-pool.yaml --verbose --split test

PYTHONPATH=/home/users/u6921098/project/VideoX/2D-TAN/lib; python moment_localization/test.py --cfg experiments/iaw/2D-TAN-conv.yaml --verbose --split test
