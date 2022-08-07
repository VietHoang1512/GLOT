export SEED=0
export PYTHONHASHSEED=$SEED

target=5
align=.05
eps=1
xi=.5
niter=15
for n_particles in 1 2 4 8 
do
    echo "TRAINING WITH ALIGNMENT $align"
    python train.py --method WDR --exp "$n_particles-particles-$align-mix-$target-target-$xi-xi-$eps-eps-$niter-niter" --xi $xi --eps $eps --niter $niter --target $target --n_particles $n_particles --beta 0. --margin .2 --align $align --epsilon .1 --label_shift .5  --s_dset_path data/image-clef/i.txt --t_dset_path data/image-clef/p.txt  --seed $SEED --dset image-clef 
    python train.py --method WDR --exp "$n_particles-particles-$align-mix-$target-target-$xi-xi-$eps-eps-$niter-niter" --xi $xi --eps $eps --niter $niter --target $target --n_particles $n_particles --beta 0. --margin .2 --align $align --epsilon .1 --label_shift .5  --s_dset_path data/image-clef/p.txt --t_dset_path data/image-clef/i.txt --seed $SEED --dset image-clef
    python train.py --method WDR --exp "$n_particles-particles-$align-mix-$target-target-$xi-xi-$eps-eps-$niter-niter" --xi $xi --eps $eps --niter $niter --target $target --n_particles $n_particles --beta 0. --margin .2 --align $align --epsilon .1 --label_shift .5  --s_dset_path data/image-clef/i.txt --t_dset_path data/image-clef/c.txt --seed $SEED --dset image-clef
    python train.py --method WDR --exp "$n_particles-particles-$align-mix-$target-target-$xi-xi-$eps-eps-$niter-niter" --xi $xi --eps $eps --niter $niter --target $target --n_particles $n_particles --beta 0. --margin .2 --align $align --epsilon .1 --label_shift .5  --s_dset_path data/image-clef/c.txt --t_dset_path data/image-clef/i.txt  --seed $SEED --dset image-clef
    python train.py --method WDR --exp "$n_particles-particles-$align-mix-$target-target-$xi-xi-$eps-eps-$niter-niter" --xi $xi --eps $eps --niter $niter --target $target --n_particles $n_particles --beta 0. --margin .2 --align $align --epsilon .1 --label_shift .5  --s_dset_path data/image-clef/c.txt --t_dset_path data/image-clef/p.txt --seed $SEED --dset image-clef
    python train.py --method WDR --exp "$n_particles-particles-$align-mix-$target-target-$xi-xi-$eps-eps-$niter-niter" --xi $xi --eps $eps --niter $niter --target $target --n_particles $n_particles --beta 0. --margin .2 --align $align --epsilon .1 --label_shift .5  --s_dset_path data/image-clef/p.txt --t_dset_path data/image-clef/c.txt --seed $SEED --dset image-clef
done


xi=2
for n_particles in 1 2 4 8 
do
    echo "TRAINING WITH ALIGNMENT $align"
    python train.py --method WDR --exp "$n_particles-particles-$align-mix-$target-target-$xi-xi-$eps-eps-$niter-niter" --xi $xi --eps $eps --niter $niter --target $target --n_particles $n_particles --beta 0. --margin .2 --align $align --epsilon .1 --label_shift .5  --s_dset_path data/image-clef/i.txt --t_dset_path data/image-clef/p.txt  --seed $SEED --dset image-clef 
    python train.py --method WDR --exp "$n_particles-particles-$align-mix-$target-target-$xi-xi-$eps-eps-$niter-niter" --xi $xi --eps $eps --niter $niter --target $target --n_particles $n_particles --beta 0. --margin .2 --align $align --epsilon .1 --label_shift .5  --s_dset_path data/image-clef/p.txt --t_dset_path data/image-clef/i.txt --seed $SEED --dset image-clef
    python train.py --method WDR --exp "$n_particles-particles-$align-mix-$target-target-$xi-xi-$eps-eps-$niter-niter" --xi $xi --eps $eps --niter $niter --target $target --n_particles $n_particles --beta 0. --margin .2 --align $align --epsilon .1 --label_shift .5  --s_dset_path data/image-clef/i.txt --t_dset_path data/image-clef/c.txt --seed $SEED --dset image-clef
    python train.py --method WDR --exp "$n_particles-particles-$align-mix-$target-target-$xi-xi-$eps-eps-$niter-niter" --xi $xi --eps $eps --niter $niter --target $target --n_particles $n_particles --beta 0. --margin .2 --align $align --epsilon .1 --label_shift .5  --s_dset_path data/image-clef/c.txt --t_dset_path data/image-clef/i.txt  --seed $SEED --dset image-clef
    python train.py --method WDR --exp "$n_particles-particles-$align-mix-$target-target-$xi-xi-$eps-eps-$niter-niter" --xi $xi --eps $eps --niter $niter --target $target --n_particles $n_particles --beta 0. --margin .2 --align $align --epsilon .1 --label_shift .5  --s_dset_path data/image-clef/c.txt --t_dset_path data/image-clef/p.txt --seed $SEED --dset image-clef
    python train.py --method WDR --exp "$n_particles-particles-$align-mix-$target-target-$xi-xi-$eps-eps-$niter-niter" --xi $xi --eps $eps --niter $niter --target $target --n_particles $n_particles --beta 0. --margin .2 --align $align --epsilon .1 --label_shift .5  --s_dset_path data/image-clef/p.txt --t_dset_path data/image-clef/c.txt --seed $SEED --dset image-clef
done