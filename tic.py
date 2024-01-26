#!/usr/bin/env python
# coding: utf-8
import os
from os.path import dirname, exists, expanduser, isdir, join, splitext
import numpy as np
from sklearn.utils import Bunch
from _base import cross_product

def load_tic(filter=None, return_X_y=False):
    base_dir = join(dirname(__file__), 'data/')
    data_filename = join(base_dir, 'ticdata2000.txt')

    header_train = ["客户子分类", "房产数量", "平均房产面积", "平均年龄", "客户主分类", "天主教", "新教", "其它教派", "非教徒", "已婚", "同居", "其它关系", "单身", "有房没有孩子", "有房有孩子", "高等学历", "中等学历", "低等学历", "地位高", "企业家", "农民", "中层管理人员", "技术工人", "非熟练工人", "A类", "B1类", "B2类", "C类", "D类", "租房", "房东", "1辆车", "2辆车", "没有车", "国家健康保险", "私人健康保险", "收入小于3万", "收入介于3万和4万5之间", "收入介于4万5和7万5之间", "收入介于7万5和12万2之间", "收入大于12万3", "平均收入", "购买力等级", "定期缴款私人第三方保险", "定期缴款第三方保险(公司)", "定期缴款第三方保险(农业)", "定期缴款汽车保单", "定期缴款货车保单", "定期缴款摩托车/踏板车保单", "定期缴款卡车保单", "定期缴款拖挂车保单", "定期缴款拖拉机保单", "定期缴款农业机械保单", "定期缴款轻便摩托车保单", "定期缴款人寿保险", "定期缴款私人意外保险保单", "定期缴款家庭意外保险保单", "定期缴款伤残保险保单", "定期缴款火灾保单", "定期缴款冲浪板保单", "定期缴款船只保单", "定期缴款自行车保单", "定期缴款财产保险保单", "定期缴款社会保障保险保单", "私人第三方保险数量", "第三方保险(公司)数量", "第三方保险(农业)数量", "汽车保单数量", "货车保单数量", "摩托车/踏板车保单数量", "卡车保单数量", "拖挂车保单数量", "拖拉机保单数量", "农业机械保单数量", "轻便摩托车保单数量", "人寿保险数量", "私人意外保险保单数量", "家庭意外保险保单数量", "伤残保险保单数量", "火灾保单数量", "冲浪板保单数量", "船只保单数量", "自行车保单数量", "财产保险保单数量", "社会保障保险保单数量", "房车保单数量"]

    header_exercise = header_train[0:(len(header_train) - 1)]
    header_physiological = [header_train[(len(header_train) - 1)]]

    data_train = np.genfromtxt(data_filename, dtype=int, delimiter="	")

    data_exercise = None
    data_physiological = None

    # 过滤训练数据
    if not filter == None:
        desicion_item = filter[0] - 1
        desicion_value = filter[1]
        filter_keep_part = filter[2]
        filter_partitions = filter[3]

        index = 0
        for i in range(data_train.shape[0]):
            row = data_train[i, :]

            item_value = row[desicion_item]
            if item_value == desicion_value:
                if index == filter_keep_part:
                    skip = False
                else:
                    skip = True

                index += 1

                if index >= filter_partitions:
                    index = 0
            else:
                skip = False

            if not skip:
                if np.all(data_exercise) == None:
                    data_exercise = np.array(row[0:(data_train.shape[1] - 1)])
                    data_physiological = np.array(row[data_train.shape[1] - 1])
                else:
                    data_exercise = np.vstack((data_exercise, np.array(row[0:(data_train.shape[1] - 1)])))
                    data_physiological = np.vstack((data_physiological, np.array(row[data_train.shape[1] - 1])))

        data_physiological = data_physiological.reshape((data_physiological.shape[0],))
    else:
        data_exercise = data_train[:, 0:(data_train.shape[1] - 1)]
        data_physiological = data_train[:, (data_train.shape[1] - 1)]

    # 增加汽车保险叉积
    PPERSAUT = 47
    APERSAUT = 68
    CP_PAT = cross_product(data_exercise[:, PPERSAUT - 1], data_exercise[:, APERSAUT - 1])
    header_exercise.append("汽车保险叉积*")
    data_exercise = np.append(data_exercise, values=CP_PAT, axis=1)

    # 增加火灾保险叉积
    PBRAND = 59
    ABRAND = 80
    CP_PAD = cross_product(data_exercise[:, PBRAND - 1], data_exercise[:, ABRAND - 1])
    header_exercise.append("火灾保险叉积*")
    data_exercise = np.append(data_exercise, values=CP_PAD, axis=1)

    with open(join(base_dir, 'TicDataDescr.txt'), encoding="Windows-1252") as f:
        descr = f.read()

    if return_X_y:
        return data_exercise, data_physiological

    return Bunch(all=data_train,
                 all_names=header_train,
                 data=data_exercise,
                 feature_names=header_exercise,
                 target=data_physiological,
                 target_names=header_physiological,
                 DESCR=descr,
                 data_filename=data_filename)
