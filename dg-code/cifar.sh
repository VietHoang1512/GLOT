export SEED=0
export PYTHONHASHSEED=$SEED
declare -A decay=( ["wrn"]=0.0005 ["resnext"]=0.0005 ["allconv"]=0.0005 ["densenet"]=0.0001)
declare -A epochs=( ["wrn"]=150 ["resnext"]=250 ["allconv"]=150 ["densenet"]=250)



lr=0.1
det=True
method=WDR
loss_type=trade
python main_cifar.py \
--seed=1 \
--weight_decay=${decay[densenet]} \
--algorithm=$method \
--model=densenet \
--dataset=cifar10 \
--epochs=${epochs[densenet]} \
--batch_size=128 \
--lr=$lr \
--logs='outputs/cifar10/'$method'/densenet/2.1.001/logs/' \
--model_path='outputs/cifar10/'$method'/densenet/2.1.001/models/' \
--model_ckpt='outputs/cifar10/pretrained/densenet/pretrained.pt' \
--deterministic=$det \
--k=2 \
--epochs_min=10 \
--eta=10.0 \
--xi=.1 \
--eps=.001 \
--n_particles=2 \
--loss_type=$loss_type &


python main_cifar.py \
--seed=1 \
--weight_decay=${decay[densenet]} \
--algorithm=$method \
--model=densenet \
--dataset=cifar100 \
--epochs=${epochs[densenet]} \
--batch_size=128 \
--lr=$lr \
--logs='outputs/cifar100/'$method'/densenet/2.1.001/logs/' \
--model_path='outputs/cifar100/'$method'/densenet/2.1.001/models/' \
--model_ckpt='outputs/cifar100/pretrained/densenet/pretrained.pt' \
--deterministic=$det \
--k=2 \
--epochs_min=10 \
--eta=10.0 \
--xi=.1 \
--eps=.001 \
--n_particles=2 \
--loss_type=$loss_type &

python main_cifar.py \
--seed=1 \
--weight_decay=${decay[densenet]} \
--algorithm=$method \
--model=densenet \
--dataset=cifar10 \
--epochs=${epochs[densenet]} \
--batch_size=128 \
--lr=$lr \
--logs='outputs/cifar10/'$method'/densenet/4.1.001/logs/' \
--model_path='outputs/cifar10/'$method'/densenet/4.1.001/models/' \
--model_ckpt='outputs/cifar10/pretrained/densenet/pretrained.pt' \
--deterministic=$det \
--k=2 \
--epochs_min=10 \
--eta=10.0 \
--xi=.1 \
--eps=.001 \
--n_particles=4 \
--loss_type=$loss_type &


python main_cifar.py \
--seed=1 \
--weight_decay=${decay[densenet]} \
--algorithm=$method \
--model=densenet \
--dataset=cifar100 \
--epochs=${epochs[densenet]} \
--batch_size=128 \
--lr=$lr \
--logs='outputs/cifar100/'$method'/densenet/4.1.001/logs/' \
--model_path='outputs/cifar100/'$method'/densenet/4.1.001/models/' \
--model_ckpt='outputs/cifar100/pretrained/densenet/pretrained.pt' \
--deterministic=$det \
--k=2 \
--epochs_min=10 \
--eta=10.0 \
--xi=.1 \
--eps=.001 \
--n_particles=4 \
--loss_type=$loss_type &

wait


