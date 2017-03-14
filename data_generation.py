import os
import cv2
import sys
import random
import argparse
import numpy as np


def removeHiddenfile(directory_list):
    if '.' in directory_list:
        directory_list.remove('.')
    if '..' in directory_list:
        directory_list.remove('.')
    if '.DS_Store' in directory_list:
        directory_list.remove('.DS_Store')
    return directory_list


def homographyGeneration(raw_image, image_path, dataPath, index, img_per_dir):

    trainDataDirPath = os.path.join(dataPath, str(index))
    if not os.path.exists(trainDataDirPath):
        os.mkdir(trainDataDirPath)

    img = cv2.resize(cv2.imread(image_path, 0),(320,240))
    # img_height = img.shape[0]
    # img_width = img.shape[1]

    #  print('height is: ', img_height)
    #  print('width is: ', img_width)

    with open(trainDataDirPath + '/homography_' + str(index) + '.txt', 'ab') as output_file:

        random_list = []
        name_suffix = 1
        i = 1
        while i < img_per_dir + 1:

            # print('<===== orginal image patch =====>')
            y_start = random.randint(32,80)
            y_end = y_start + 128
            x_start = random.randint(32,160)
            x_end = x_start + 128

            y_1 = y_start
            x_1 = x_start
            y_2 = y_end
            x_2 = x_start
            y_3 = y_end
            x_3 = x_end
            y_4 = y_start
            x_4 = x_end
            # print(y_start)
            # print(y_end)
            # print(x_start)
            # print(x_end)
            # print('\n')
            img_patch = img[y_start:y_end, x_start:x_end]
            # cv2.imshow('patch', img_patch)
            #  cv2.waitKey(0)
            # print('<===== perburbed image patch =====>')
            y_1_offset = random.randint(-32,32)
            x_1_offset = random.randint(-32,32)
            y_2_offset = random.randint(-32,32)
            x_2_offset = random.randint(-32,32)

            y_3_offset = random.randint(-32,32)
            x_3_offset = random.randint(-32,32)
            y_4_offset = random.randint(-32,32)
            x_4_offset = random.randint(-32,32)

            y_1_p = y_1 + y_1_offset
            x_1_p = x_1 + x_1_offset
            y_2_p = y_2 + y_2_offset
            x_2_p = x_2 + x_2_offset
            y_3_p = y_3 + y_3_offset
            x_3_p = x_3 + x_3_offset
            y_4_p = y_4 + y_4_offset
            x_4_p = x_4 + x_4_offset
            # print(y_p_start)
            # print(y_p_end)
            # print(x_p_start)
            # print(x_p_end)
            # img_patch_perturb = img[y_p_start:y_p_end, x_p_start:x_p_end]
            # cv2.imshow('p_patch', img_patch_perturb)
            # cv2.waitKey(0)
            pts_img_patch = np.array([[y_1,x_1],[y_2,x_2],[y_3,x_3],[y_4,x_4]]).astype(np.float32)
            pts_img_patch_perturb = np.array([[y_1_p,x_1_p],[y_2_p,x_2_p],[y_3_p,x_3_p],[y_4_p,x_4_p]]).astype(np.float32)
            h,status = cv2.findHomography(pts_img_patch, pts_img_patch_perturb, cv2.RANSAC)
            # print(h)
            # print(status)
            img_perburb = cv2.warpPerspective(img, h, (320,240))
            # print(img_perburb2.shape)
            # cv2.imshow('perburb', img_perburb)
            #  cv2.waitKey(0)
            img_perburb_patch = img_perburb[y_start:y_end, x_start:x_end]
            if not [y_1,x_1,y_2,x_2,y_3,x_3,y_4,x_4] in random_list:
                random_list.append([y_1,x_1,y_2,x_2,y_3,x_3,y_4,x_4])

                h_4pt = np.array([y_1_p,x_1_p,y_2_p,x_2_p,y_3_p,x_3_p,y_4_p,x_4_p])

                img_patch_path = os.path.join(trainDataDirPath, (raw_image.split('.')[0] + '_' + str(name_suffix) + '_1' +'.jpg'))
                cv2.imwrite(img_patch_path, img_patch)
                img_perburb_patch_path = os.path.join(trainDataDirPath, (raw_image.split('.')[0] + '_' + str(name_suffix) + '_2' +'.jpg'))
                cv2.imwrite(img_perburb_patch_path, img_perburb_patch)
                name_suffix += 1
                np.savetxt(output_file, h_4pt)
                #output_file.write('\n')
                i += 1


def dataCollection(rawDataPath, dataPath, img_number,img_per_dir):
    raw_image_list = removeHiddenfile(os.listdir(rawDataPath))
    index = 1
    for raw_image in raw_image_list:
        print(str(index) + '\n')
        raw_image_path = os.path.join(rawDataPath,raw_image)
        homographyGeneration(raw_image, raw_image_path, dataPath, index, img_per_dir)
        if index == img_number + 1:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rawTrainDataPath', type=str, default='/mnt/backup/Datasets/ms-coco/train2014', help='The raw data path.')
    parser.add_argument('--trainDataPath', type=str, default='/home/runze/DeepHomography/data/trainData/', help='The training data path')
    parser.add_argument('--trainDataNumber', type=int, default=8000, help='The data size for traininig')

    # parser.add_argument('--rawValDataPath', type=str, default='/mnt/backup/Datasets/ms-coco/val2014', help='The raw data path.')
    # parser.add_argument('--valDataPath', type=str, default='/home/runze/DeepHomography/data/valData/', help='The training data path')
    # parser.add_argument('--valDataNumber', type=int, default=2000, help='The data size for validate tuning')

    parser.add_argument('--rawTestDataPath', type=str, default='/mnt/backup/Datasets/ms-coco/test2014', help='The raw data path.')
    parser.add_argument('--testDataPath', type=str, default='/home/runze/DeepHomography/data/testData/', help='The training data path')
    parser.add_argument('--testDataNumber', type=int, default=800, help='The data size for testing')

    print('<==================== Loading raw data ===================>\n')
    args = parser.parse_args()
    print('<================= Generating homography =================>\n')
    dataCollection(args.rawTrainDataPath, args.trainDataPath, args.trainDataNumber, 4)

    dataCollection(args.rawValDataPath, args.valDataPath, args.valDataNumber, 4)

    dataCollection(args.rawTestDataPath, args.testDataPath, args.testDataNumber, 1)

    print('<=================== Finishing data ===================>')


if __name__ == '__main__':
    main()
