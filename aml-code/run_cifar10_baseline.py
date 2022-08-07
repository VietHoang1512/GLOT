import subprocess 
import numpy as np 
import os 
import sys

os.chdir('./')
ST = 'python '

stand = dict()
conf = dict()

stand = dict()
stand['ds'] = 'cifar10' 
stand['bs'] = 128
stand['defense'] = 'none'
stand['model'] = 'resnet18'
stand['epsilon'] = 0.031
stand['projecting'] = True
stand['trades_beta'] = 6.0
stand['eval_multi'] = False

# conf['pgd_train'] = stand.copy()
# conf['pgd_train']['defense'] = 'pgd_train'
# conf['pgd_train']['trades_beta'] = 1.0 
# conf['pgd_train']['alpha'] = 0.0 

# conf['trades_train'] = stand.copy()
# conf['trades_train']['defense'] = 'trades_train'
# conf['trades_train']['trades_beta'] = 6.0 

# conf['mart_train'] = stand.copy()
# conf['mart_train']['defense'] = 'mart_train'
# conf['mart_train']['trades_beta'] = 5.0 

# RUNNING, FIT, GPU 4,5
# conf['test_1'] = stand.copy()
# conf['test_1']['defense'] = 'pgd_svgd_ce'
# conf['test_1']['trades_beta'] = 1.0 
# conf['test_1']['alpha'] = 0.0 
# conf['test_1']['wcom'] = 0.0
# conf['test_1']['wkl'] = 1.0 
# conf['test_1']['num_particles'] = 2
# conf['test_1']['dist'] = 'cosine'

# RUNNING, FIT, GPU 6, 7
# conf['test_2'] = stand.copy()
# conf['test_2']['defense'] = 'pgd_svgd_ce'
# conf['test_2']['trades_beta'] = 1.0 
# conf['test_2']['alpha'] = 0.0 
# conf['test_2']['wcom'] = 1.0 
# conf['test_2']['wkl'] = 0.0 
# conf['test_2']['num_particles'] = 2
# conf['test_2']['dist'] = 'cosine'

# RUNNING, FIT, GPU 0, 1
# conf['test_3'] = stand.copy()
# conf['test_3']['defense'] = 'pgd_svgd_ce'
# conf['test_3']['trades_beta'] = 1.0 
# conf['test_3']['alpha'] = 0.0 
# conf['test_3']['wcom'] = 1.0 
# conf['test_3']['wkl'] = 1.0 
# conf['test_3']['num_particles'] = 2
# conf['test_3']['dist'] = 'cosine'

# RUNNING, FIT, GPU 2, 3
conf['test_4'] = stand.copy()
conf['test_4']['defense'] = 'pgd_svgd_ce'
conf['test_4']['trades_beta'] = 1.0 
conf['test_4']['alpha'] = 0.0 
conf['test_4']['wcom'] = 1.0 
conf['test_4']['wkl'] = 1.0 
conf['test_4']['num_particles'] = 4
conf['test_4']['dist'] = 'cosine'
skip = ['_', '_', '_', '_']

progs = [
	'02d_adversarial_training.py ',
    '02e_evaluate_robustness.py ',
]

for k in list(conf.keys()):
	if k in skip: 
		continue

	for chST in progs: 

		exp = conf[k]
		sub = ' '.join(['--{}={}'.format(t, exp[t]) for t in exp.keys()])
		print(sub)
		subprocess.call([ST + chST + sub], shell=True)