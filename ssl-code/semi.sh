export PYTHONPATH=$PWD

python train_semi.py --method=trades_svgd_ot --n=4 --dataset=cifar10 --model=cnn13 --num_label=1000 --batch_size=128 --num_epochs=600 --beta=1.0 --gamma=0.1
python train_semi.py --method=trades_svgd_ot --n=4 --dataset=cifar10 --model=cnn13 --num_label=4000 --batch_size=128 --num_epochs=600 --beta=10.0 --gamma=1.0
