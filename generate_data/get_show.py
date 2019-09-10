#!/usr/local/bin/ipython
import imageio
import random
import glob
import matplotlib.pyplot as plt
import config as cfg
from textwrap import wrap
import argparse


def get_data(aligned=False):
    mode = 'aligned' if aligned else 'normal'

    txt_files = glob.glob('../data/MultiLabelAU/{}/*/test.txt'.format(mode))
    lines = []
    replace = 'Faces_256' if not aligned else 'Faces_aligned_256'
    for txt in txt_files:
        lines.extend([
            line.strip().replace('Sequences', replace)
            for line in sorted(open(txt).readlines())
        ])
    lines.sort()
    return lines


def imshow_data(line):
    img = line.split(' ')[0]
    labels = [int(i) for i in line.split(' ')[1:]]
    aus = [
        cfg.AUs_name_es[i] for i in range(len(cfg.AUs_name_es))
        if labels[i] == 1
    ]
    # print(str(labels))
    # fig = plt.figure(figsize=(15, 15))
    plt.imshow(imageio.imread(img))
    title = plt.title('\n'.join(wrap(' - '.join(aus), 60)))
    title.set_fontsize(24)
    title.set_fontweight('bold')
    plt.axis('off')
    # title.set_y(0.45)
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--random', action='store_true', default=False)
    parser.add_argument('--aligned', action='store_true', default=False)
    args = parser.parse_args()
    all_data = get_data(args.aligned)
    rand_idx = 0
    try:
        while True:
            if args.random:
                rand_idx = random.randint(0, len(all_data))
            else:
                rand_idx += 1
            imshow_data(all_data[rand_idx])
    except KeyboardInterrupt:
        pass
