import os
import numpy as np

source_path = 'pascal_valid'
lst_file = 'pascal24_valid.lst'

lst = open(lst_file, "w")

counter = 0
for f_name in os.listdir(source_path):
    if '.jpg' in f_name:
        try:
            old_label_name = f_name.split('.')[0] + '.txt'
            label = np.loadtxt(os.path.join(source_path, old_label_name), dtype='float')
            L = [None]*5
            L[0] = int(label[0])
            L[1] = label[1] - label[3]/2
            L[2] = label[2] - label[4]/2
            L[3] = label[1] + label[3]/2
            L[4] = label[2] + label[4]/2

            lst.write('\n%d\t2\t5' % counter)
            lst.write('\t%d\t%.4f\t%.4f\t%.4f\t%.4f' % (
                L[0], L[1], L[2], L[3], L[4]))
            lst.write('\t' + os.path.join(source_path, f_name))

            counter += 1
            if counter % 5000 == 0:
                print(counter)
        except:
            pass
lst.close()
