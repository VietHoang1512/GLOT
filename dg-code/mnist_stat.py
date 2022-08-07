import os
import re
import numpy as np
output_dir = 'outputs/mnist/WDR_trade-stardard_4_particles_.3_.001'
n_domains = 4
best_accs = []
domain_accs = np.zeros((len(os.listdir(output_dir)),n_domains))
for i, subdir in enumerate(sorted(os.listdir(output_dir))):
    best_test_file = os.path.join(output_dir, subdir, 'best_test.txt')
    with open(best_test_file, "r") as f:
        best_test = f.read()
    best_acc = float(re.findall('best test accuracy:((?:\d+|\.)+)', best_test)[-1])
    best_accs.append(best_acc)
    print(subdir, best_acc)
    for j in range(n_domains):
        domain_file = os.path.join(output_dir, subdir, 'test_index_%d.txt' % j)
        with open(domain_file, "r") as f:
            domain = f.read()
        domain_acc = float(re.findall('accuracy:((?:\d+|\.)+) best!', domain)[-1])
        print('\tdomain %d: %.4f' % (j, domain_acc))
        domain_accs[i][j] = domain_acc
print("Best accs:", f"{100*np.mean(best_accs):.2f}", "±", f"{100*np.std(best_accs):.2f}")
mean_domain_accs = []
std_domain_accs = []
for j in range(n_domains):
    mean_domain_acc = 100*np.mean(domain_accs[:,j])
    std_domain_acc = 100*np.std(domain_accs[:,j])
    print("Domain acc", j, f"{mean_domain_acc:.2f}", "±", f"{std_domain_acc:.2f}")
    mean_domain_accs.append(mean_domain_acc)
    std_domain_accs.append(std_domain_acc)
print("Average:", f"{np.mean(mean_domain_accs):.2f}", "±", f"{np.mean(std_domain_accs):.2f}")