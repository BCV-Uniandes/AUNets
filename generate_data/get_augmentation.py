import os
import imageio
import numpy as np
import argparse
from jittering import jitter_points
import tqdm
import skimage.transform
import glob
import shutil
img_size = 256


def imshow(image):
    import cv2 as cv
    if isinstance(image, str):
        image = cv.imread(image)
    cv.startWindowThread()
    cv.namedWindow('Image', cv.WINDOW_NORMAL)
    cv.imshow('Image', image)


def extract_Jittering(img, file_jitter, rect_face, jit_idx):
    file_jitter_resized = file_jitter
    folder_jitter = '/'.join(file_jitter_resized.split('/')[:-1])
    if not os.path.isdir(folder_jitter):
        os.makedirs(folder_jitter)
    if os.path.isfile(file_jitter_resized):
        return file_jitter_resized
    try:
        # Since rect_face comes in form of img[rect[0]:rect[1],
        # rect[2]:rect[3]]
        rect_face = [rect_face[i] for i in [2, 0, 3, 1]]
    except BaseException:
        return file_jitter_resized
    rect_face_j = jitter_points(rect_face, img.shape, jit_idx)

    img_cropped = img[rect_face_j[1]:rect_face_j[3], rect_face_j[0]:
                      rect_face_j[2]]
    try:
        img_cropped_resized = (
            skimage.transform.resize(img_cropped,
                                     (img_size, img_size)) * 255).astype(
                                         np.uint8)
        imageio.imwrite(file_jitter_resized, img_cropped_resized)
    except BaseException:
        return file_jitter_resized
    return file_jitter_resized


def extract_Mirroring(face_file):
    elicited = face_file.split('/')[-2]
    new_file = face_file.replace(elicited, elicited + '_flip')
    folder = os.path.dirname(new_file)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    if not os.path.isfile(new_file):
        img_m = imageio.imread(face_file)[:, ::-1]
        try:
            imageio.imwrite(new_file, img_m)
        except BaseException:
            return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Data augmentation with jittering. Optical Flow option.')
    parser.add_argument('--OF', action='store_true', default=False)
    args = parser.parse_args()

    folder_root = '/home/afromero/datos/Databases/BP4D/Sequences'
    Data = sorted(glob.glob(folder_root + '/*/*/*.jpg'))
    data_img = Data
    data_face = [i.replace('Sequences', 'Faces') for i in Data]
    data_rect = [
        i.replace('Sequences', 'BBOX').replace('jpg', 'txt') for i in Data
    ]
    aug = 6  # 0--6 jittering

    for idx in tqdm.tqdm(
            range(len(data_img)),
            desc='Extracting %d jittering per image' % (aug)):

        if not os.path.isfile(data_face[idx]):
            continue
        name_faces = 'Faces_' + str(img_size)
        face_file = data_face[idx].replace('Faces', name_faces)
        extract_Mirroring(face_file)

        if args.OF:
            name_faces_of = 'Faces_Flow_' + str(img_size)
            face_file_of = data_face[idx].replace('Faces', name_faces_of)
            if not os.path.isfile(face_file_of):
                len_string_previous = len(
                    os.path.basename(face_file_of).split('.')[0])
                previous = str(
                    int(os.path.basename(face_file_of).split('.')[0]) -
                    1).zfill(len_string_previous)
                previous_file = os.path.join(
                    os.path.dirname(face_file_of), previous + '.jpg')
                shutil.copyfile(previous_file, face_file_of)
            extract_Mirroring(face_file_of)

        flag = True
        flag_OF = True
        for jitter_idx in range(aug):
            file_jitter = face_file.replace(
                name_faces, name_faces + '/Jitter/Jitter_' + str(jitter_idx))

            if not os.path.isfile(file_jitter):
                if flag:
                    img_org = imageio.imread(data_img[idx])
                    flag = False
                face_rect = [
                    int(i.strip()) for i in open(data_rect[idx]).readlines()
                ]
                img_j = extract_Jittering(img_org, file_jitter, face_rect,
                                          jitter_idx)
                extract_Mirroring(img_j)

            if args.OF:
                file_jitter_of = face_file.replace(
                    name_faces_of,
                    name_faces_of + '/Jitter/Jitter_' + str(jitter_idx))
                if not os.path.isfile(file_jitter_of):
                    if flag_OF:
                        org_of = data_img[idx].replace('Sequences',
                                                       'Sequences_Flow')
                        if not os.path.isfile(org_of):
                            len_string_previous = len(
                                os.path.basename(org_of).split('.')[0])
                            previous = str(
                                int(os.path.basename(org_of).split('.')[0]) -
                                1).zfill(len_string_previous)
                            org_of = os.path.join(
                                os.path.dirname(org_of), previous + '.jpg')
                        img_org_of = imageio.imread(org_of)
                        flag_OF = False
                    face_rect = [
                        int(i.strip())
                        for i in open(data_rect[idx]).readlines()
                    ]
                    img_j = extract_Jittering(img_org_of, file_jitter_of,
                                              face_rect, jitter_idx)
                    extract_Mirroring(img_j)
