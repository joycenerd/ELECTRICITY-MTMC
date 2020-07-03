set -e
dir=$(dirname $(readlink -fn --  $0))
python ./identifier/train.py --train_sets Aic --test_set Aic -a resnet101 --save-dir models/resnet101-Aic --root ./datasets
