#!/usr/bin/env ipython
import tqdm
import argparse
import config as cfg
import os
import numpy as np
import sys
sys.path.insert(0, '..')


def create_dir(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Txt file with path_to_image and 12 different AUs to LMDB')
    parser.add_argument(
        '--mode',
        type=str,
        default='normal',
        help='Mode: Training/Test (default: normal)')
    parser.add_argument(
        '--AU',
        type=str,
        default='all',
        help='Crossvalidation fold (default: all)')
    parser.add_argument(
        '--fold',
        type=str,
        default='all',
        help='Crossvalidation fold (default: all)')
    parser.add_argument(
        '--dataset',
        type=str,
        default='BP4D',
        help='Crossvalidation fold (default: BP4D)')
    args = parser.parse_args()

    FOLDS = [args.fold] if args.fold.isdigit() else [0, 1, 2]
    AUs = cfg.AUs if args.AU == 'all' else [args.AU]

    # print("Mode: "+mode+'\nFold: '+str(fold))
    for fold in FOLDS:
        txt_path = os.path.join(cfg.TXT_PATH, args.dataset, args.mode,
                                'fold_' + str(fold))
        train_full_txt = os.path.join(txt_path, 'train.txt')
        lines = [line.strip() for line in open(train_full_txt).readlines()]

        val_full_txt = os.path.join(txt_path, 'val.txt')
        lines_val = [line.strip() for line in open(val_full_txt).readlines()]

        test_full_txt = os.path.join(txt_path, 'test.txt')
        lines_test = [line.strip() for line in open(test_full_txt).readlines()]

        for AU in AUs:
            txt_folder_au = os.path.join(txt_path, 'AU' + str(AU).zfill(2))
            create_dir(txt_folder_au)
            train_txt = os.path.join(txt_folder_au, 'train.txt')
            val_txt = os.path.join(txt_folder_au, 'val.txt')
            test_txt = os.path.join(txt_folder_au, 'test.txt')

            pos_AU = np.where(np.array(cfg.AUs) == AU)[0][0]

            positive_check = 0
            negative_check = 0
            new_lines = {0: [], 1: []}

            for line in lines:
                img = line.split(' ')[0]
                img = img.replace('Faces', 'Faces_256')
                if args.dataset == 'aligned':
                    img = img.replace('Faces', 'Faces_aligned')
                img_of = img.replace(
                    'Faces_256',
                    'Faces_Flow_256') if args.mode == 'normal' else img
                # if not os.path.isfile(img) or os.stat(img).st_size==0:
                # continue
                if args.mode == 'normal' and (not os.path.isfile(img)
                                              or not os.path.isfile(img_of)):
                    continue
                au = line.split(' ')[pos_AU + 1]
                if int(au) == 1:
                    positive_check += 1
                if int(au) == 0:
                    negative_check += 1
                new_lines[int(au)].append(img + ' ' + au)
            check_idx = positive_check < negative_check
            check = max(positive_check, negative_check)
            augmentation = int(
                (check - min(positive_check, negative_check)) /
                min(positive_check,
                    negative_check)) + 1 if args.mode != 'aligned' else 0
            string_ = 'Mode: {} | Fold: {} | AU: {} | +: {} | -: {} | \
                AUG: {}'.format(args.mode, fold,
                                str(AU).zfill(2), positive_check,
                                negative_check, augmentation)
            # print('\n')
            count = len(new_lines[check_idx])
            Data = []
            tqdm_bar = tqdm.tqdm(range(augmentation), desc=string_)
            for i in tqdm_bar:
                if count == check:
                    break
                for line in new_lines[check_idx]:
                    if check_idx == 0:
                        tqdm_bar.set_description(
                            string_ + ' | ++: {} | --:{}'.format(
                                len(new_lines[0]) +
                                len(Data), len(new_lines[1])))
                    else:
                        tqdm_bar.set_description(
                            string_ + ' | ++: {} | --:{}'.format(
                                len(new_lines[0]),
                                len(new_lines[1]) + len(Data)))

                    if count == check:
                        break
                    img = line.split(' ')[0]
                    au = int(line.split(' ')[1])
                    img_j = img.replace('Faces_256',
                                        'Faces_256/Jitter/Jitter_{}'.format(i))
                    if not os.path.isfile(img_j):
                        continue  # or os.stat(img_jj).st_size==0: continue
                    Data.append(img_j + ' ' + str(au))
                    count += 1
            Data = Data + new_lines[0] + new_lines[1]
            Data = sorted(Data)
            Data_flip = [
                line.replace(
                    line.split('/')[-2],
                    line.split('/')[-2] + '_flip') for line in Data
            ]
            Data = sorted(Data + Data_flip)

            new_txt = open(train_txt, 'w')
            for line in Data:
                new_txt.writelines(line + '\n')
            new_txt.close()

            new_txt = open(val_txt, 'w')
            for line in lines_val:
                if args.dataset == 'aligned':
                    img = line.split(' ')[0].replace('Faces', 'Faces_aligned')
                else:
                    img = line.split(' ')[0]
                img = img.replace('Faces', 'Faces_256')
                new_txt.writelines(img + ' ' + line.split(' ')[pos_AU + 1] +
                                   '\n')
            new_txt.close()

            new_txt = open(test_txt, 'w')
            for line in lines_test:
                if args.dataset == 'aligned':
                    img = line.split(' ')[0].replace('Faces', 'Faces_aligned')
                else:
                    img = line.split(' ')[0]
                img = img.replace('Faces', 'Faces_256')
                new_txt.writelines(img + ' ' + line.split(' ')[pos_AU + 1] +
                                   '\n')
            new_txt.close()

        print('')
