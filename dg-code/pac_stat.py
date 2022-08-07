import os
import re
import numpy as np
output_dir = 'outputs/pacs_full_final_new/WDR'
categories = sorted(os.listdir(output_dir))
categories = [os.path.join(output_dir, category) for category in categories]
print(categories)
for i, exp in enumerate(sorted(os.listdir(categories[1]))):
    try:
        print("*"*50)
        best_accs = []
        for domain in categories:
            fn = os.path.join(domain, exp)
            best_test_file = os.path.join(fn, 'dg_test.txt')
            # print("Reading", best_test_file)
            with open(best_test_file, "r") as f:
                best_test = f.read()
            # print("OK")
            best_acc = float(re.findall('accuracy:((?:\d+|\.)+) best!', best_test)[-1])
            print(fn, best_acc)
            best_accs.append(best_acc)
        print("Best accs:", f"{100*np.mean(best_accs):.2f}", "±", f"{100*np.std(best_accs):.2f}")
    except:
        pass