seen_index=0
lr=0.0001
lps=15000
step_size=15000
det=True
method=WDR
loss_type=trade
n_particles=4
xi=.3
eps=.001

for n_particles in 1 2 4 8
do 
    python main_mnist.py \
        --seed=1 \
        --algorithm=$method \
        --lr=$lr \
        --num_classes=10 \
        --test_every=100 \
        --logs='outputs/mnist/'$method'_'$loss_type'-stardard_'$n_particles'_particles_'$xi'_'$eps'/1/' \
        --batch_size=32 \
        --model_path='outputs/mnist/'$method'_'$loss_type'-stardard_'$n_particles'_particles_'$xi'_'$eps'/1/' \
        --seen_index=$seen_index \
        --loops_train=$lps \
        --loops_min=100 \
        --step_size=$step_size \
        --deterministic=$det \
        --k=3 \
        --eta=10.0 \
        --loops_adv=15 \
        --gamma=1 \
        --xi=$xi \
        --eps=$eps \
        --n_particles=$n_particles \
        --loss_type=$loss_type &

    python main_mnist.py \
        --seed=2 \
        --algorithm=$method \
        --lr=$lr \
        --num_classes=10 \
        --test_every=100 \
        --logs='outputs/mnist/'$method'_'$loss_type'-stardard_'$n_particles'_particles_'$xi'_'$eps'/2/' \
        --batch_size=32 \
        --model_path='outputs/mnist/'$method'_'$loss_type'-stardard_'$n_particles'_particles_'$xi'_'$eps'/2/' \
        --seen_index=$seen_index \
        --loops_train=$lps \
        --loops_min=100 \
        --step_size=$step_size \
        --deterministic=$det \
        --k=3 \
        --eta=10.0 \
        --loops_adv=15 \
        --gamma=1 \
        --xi=$xi \
        --eps=$eps \
        --n_particles=$n_particles \
        --loss_type=$loss_type &
        
    python main_mnist.py \
        --seed=3 \
        --algorithm=$method \
        --lr=$lr \
        --num_classes=10 \
        --test_every=100 \
        --logs='outputs/mnist/'$method'_'$loss_type'-stardard_'$n_particles'_particles_'$xi'_'$eps'/3/' \
        --batch_size=32 \
        --model_path='outputs/mnist/'$method'_'$loss_type'-stardard_'$n_particles'_particles_'$xi'_'$eps'/3/' \
        --seen_index=$seen_index \
        --loops_train=$lps \
        --loops_min=100 \
        --step_size=$step_size \
        --deterministic=$det \
        --k=3 \
        --eta=10.0 \
        --loops_adv=15 \
        --gamma=1 \
        --xi=$xi \
        --eps=$eps \
        --n_particles=$n_particles \
        --loss_type=$loss_type &
        
    python main_mnist.py \
        --seed=4 \
        --algorithm=$method \
        --lr=$lr \
        --num_classes=10 \
        --test_every=100 \
        --logs='outputs/mnist/'$method'_'$loss_type'-stardard_'$n_particles'_particles_'$xi'_'$eps'/4/' \
        --batch_size=32 \
        --model_path='outputs/mnist/'$method'_'$loss_type'-stardard_'$n_particles'_particles_'$xi'_'$eps'/4/' \
        --seen_index=$seen_index \
        --loops_train=$lps \
        --loops_min=100 \
        --step_size=$step_size \
        --deterministic=$det \
        --k=3 \
        --eta=10.0 \
        --loops_adv=15 \
        --gamma=1 \
        --xi=$xi \
        --eps=$eps \
        --n_particles=$n_particles \
        --loss_type=$loss_type &
        
    python main_mnist.py \
        --seed=5 \
        --algorithm=$method \
        --lr=$lr \
        --num_classes=10 \
        --test_every=100 \
        --logs='outputs/mnist/'$method'_'$loss_type'-stardard_'$n_particles'_particles_'$xi'_'$eps'/5/' \
        --batch_size=32 \
        --model_path='outputs/mnist/'$method'_'$loss_type'-stardard_'$n_particles'_particles_'$xi'_'$eps'/5/' \
        --seen_index=$seen_index \
        --loops_train=$lps \
        --loops_min=100 \
        --step_size=$step_size \
        --deterministic=$det \
        --k=3 \
        --eta=10.0 \
        --loops_adv=15 \
        --gamma=1 \
        --xi=$xi \
        --eps=$eps \
        --n_particles=$n_particles \
        --loss_type=$loss_type &
        
    python main_mnist.py \
        --seed=6 \
        --algorithm=$method \
        --lr=$lr \
        --num_classes=10 \
        --test_every=100 \
        --logs='outputs/mnist/'$method'_'$loss_type'-stardard_'$n_particles'_particles_'$xi'_'$eps'/6/' \
        --batch_size=32 \
        --model_path='outputs/mnist/'$method'_'$loss_type'-stardard_'$n_particles'_particles_'$xi'_'$eps'/6/' \
        --seen_index=$seen_index \
        --loops_train=$lps \
        --loops_min=100 \
        --step_size=$step_size \
        --deterministic=$det \
        --k=3 \
        --eta=10.0 \
        --loops_adv=15 \
        --gamma=1 \
        --xi=$xi \
        --eps=$eps \
        --n_particles=$n_particles \
        --loss_type=$loss_type &
        
    python main_mnist.py \
        --seed=7 \
        --algorithm=$method \
        --lr=$lr \
        --num_classes=10 \
        --test_every=100 \
        --logs='outputs/mnist/'$method'_'$loss_type'-stardard_'$n_particles'_particles_'$xi'_'$eps'/7/' \
        --batch_size=32 \
        --model_path='outputs/mnist/'$method'_'$loss_type'-stardard_'$n_particles'_particles_'$xi'_'$eps'/7/' \
        --seen_index=$seen_index \
        --loops_train=$lps \
        --loops_min=100 \
        --step_size=$step_size \
        --deterministic=$det \
        --k=3 \
        --eta=10.0 \
        --loops_adv=15 \
        --gamma=1 \
        --xi=$xi \
        --eps=$eps \
        --n_particles=$n_particles \
        --loss_type=$loss_type &
        
    python main_mnist.py \
        --seed=8 \
        --algorithm=$method \
        --lr=$lr \
        --num_classes=10 \
        --test_every=100 \
        --logs='outputs/mnist/'$method'_'$loss_type'-stardard_'$n_particles'_particles_'$xi'_'$eps'/8/' \
        --batch_size=32 \
        --model_path='outputs/mnist/'$method'_'$loss_type'-stardard_'$n_particles'_particles_'$xi'_'$eps'/8/' \
        --seen_index=$seen_index \
        --loops_train=$lps \
        --loops_min=100 \
        --step_size=$step_size \
        --deterministic=$det \
        --k=3 \
        --eta=10.0 \
        --loops_adv=15 \
        --gamma=1 \
        --xi=$xi \
        --eps=$eps \
        --n_particles=$n_particles \
        --loss_type=$loss_type &
        
    python main_mnist.py \
        --seed=9 \
        --algorithm=$method \
        --lr=$lr \
        --num_classes=10 \
        --test_every=100 \
        --logs='outputs/mnist/'$method'_'$loss_type'-stardard_'$n_particles'_particles_'$xi'_'$eps'/9/' \
        --batch_size=32 \
        --model_path='outputs/mnist/'$method'_'$loss_type'-stardard_'$n_particles'_particles_'$xi'_'$eps'/9/' \
        --seen_index=$seen_index \
        --loops_train=$lps \
        --loops_min=100 \
        --step_size=$step_size \
        --deterministic=$det \
        --k=3 \
        --eta=10.0 \
        --loops_adv=15 \
        --gamma=1 \
        --xi=$xi \
        --eps=$eps \
        --n_particles=$n_particles \
        --loss_type=$loss_type &
        
    python main_mnist.py \
        --seed=10 \
        --algorithm=$method \
        --lr=$lr \
        --num_classes=10 \
        --test_every=100 \
        --logs='outputs/mnist/'$method'_'$loss_type'-stardard_'$n_particles'_particles_'$xi'_'$eps'/10/' \
        --batch_size=32 \
        --model_path='outputs/mnist/'$method'_'$loss_type'-stardard_'$n_particles'_particles_'$xi'_'$eps'/10/' \
        --seen_index=$seen_index \
        --loops_train=$lps \
        --loops_min=100 \
        --step_size=$step_size \
        --deterministic=$det \
        --k=3 \
        --eta=10.0 \
        --loops_adv=15 \
        --gamma=1 \
        --xi=$xi \
        --eps=$eps \
        --n_particles=$n_particles \
        --loss_type=$loss_type &
    wait
done

