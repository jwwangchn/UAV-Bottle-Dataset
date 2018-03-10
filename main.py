# -*- coding: UTF-8 -*-
import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from xml.dom.minidom import Document
import cPickle as pickle
import matplotlib.pyplot as plt
import shutil
import scipy.io as scio


def parse_rbbox(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    root = tree.getroot()
    objects = []
    image_size = root.find('size')
    width = int(image_size.find('width').text)
    height = int(image_size.find('height').text)
    depth = int(image_size.find('depth').text)
    segmented = int(root.find('segmented').text)
    for object in root.findall('object'):
        obj_struct = {}
        obj_struct['size'] = [width, height, depth]
        obj_struct['segmented'] = segmented
        obj_struct['name'] = object.find('name').text
        obj_struct['pose'] = object.find('pose').text
        obj_struct['truncated'] = int(object.find('truncated').text)
        obj_struct['difficult'] = int(object.find('difficult').text)
        robndbox = object.find('robndbox')
        cx = int(float(robndbox.find('cx').text))
        cy = int(float(robndbox.find('cy').text))
        w = int(float(robndbox.find('w').text))
        h = int(float(robndbox.find('h').text))
        angle = float(robndbox.find('angle').text)
        angle = int(-angle * 180.0 / np.pi)
        obj_struct['bbox'] = [cx, cy, h, w, angle]
        objects.append(obj_struct)
    return objects


def parse_bbox(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    root = tree.getroot()
    objects = []
    path = root.find('path').text
    filename_ = root.find('filename').text
    image_size = root.find('size')
    width = int(image_size.find('width').text)
    height = int(image_size.find('height').text)
    depth = int(image_size.find('depth').text)
    segmented = int(root.find('segmented').text)
    for object in root.findall('object'):
        obj_struct = {}
        obj_struct['path'] = path
        obj_struct['filename'] = filename_
        obj_struct['size'] = [width, height, depth]
        obj_struct['segmented'] = segmented
        obj_struct['name'] = object.find('name').text
        obj_struct['pose'] = object.find('pose').text
        obj_struct['truncated'] = int(object.find('truncated').text)
        obj_struct['difficult'] = int(object.find('difficult').text)
        bndbox = object.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))

        obj_struct['bbox'] = [xmin, ymin, xmax, ymax]
        objects.append(obj_struct)
    return objects


def check_image_annotations(delete_files=False):
    """
    用于检查原始图像和标注文件的数量
    delete_files: 是否删除不匹配的文件
    :return:
    """
    categories = ['1_Sand', '2_Lawn', '3_Bush', '4_Land', '5_Step', '6_Mixture', '7_Ground', '8_Playground']
    root_dir = 'E:/jwwangchn/Data/UAV-Bottle/UAV-Bottle-V3.0.0/Categroies'

    for category in categories:
        image_path = os.path.join(root_dir, category)
        annotation_path = os.path.join(root_dir, category + '_Annotations')

        for image_name in os.listdir(image_path):
            image = os.path.join(image_path, image_name)
            annotation = os.path.join(annotation_path, image_name.split('.')[0] + '.xml')

            if not os.path.exists(annotation):
                if delete_files:
                    os.remove(image)
                print
                image_name


def rename_two_phase_file(root_path, phase):
    categories = ['1_Sand', '2_Lawn', '3_Bush', '4_Land', '5_Step', '6_Mixture', '7_Ground', '8_Playground']

    for category in categories:
        image_path = os.path.join(root_path, category)
        annotation_path = os.path.join(root_path, category + '_Annotations')
        image_lists = os.listdir(image_path)
        for image_name in image_lists:
            no_extension_name = image_name.split('.')[0]
            image = os.path.join(image_path, image_name)
            annotation = os.path.join(annotation_path, no_extension_name + '.xml')

            if phase == '1':
                new_image = os.path.join(image_path, no_extension_name + '_1' + '.jpg')
                new_annotation = os.path.join(annotation_path, no_extension_name + '_1' + '.xml')
            elif phase == '2':
                new_image = os.path.join(image_path, no_extension_name + '_2' + '.jpg')
                new_annotation = os.path.join(annotation_path, no_extension_name + '_2' + '.xml')
            os.rename(image, new_image)
            os.rename(annotation, new_annotation)


def num_each_category(root_path):
    categories_dict = {'1_Sand': 0, '2_Lawn': 0, '3_Bush': 0, '4_Land': 0, '5_Step': 0, '6_Mixture': 0, '7_Ground': 0,
                       '8_Playground': 0}
    for file_name in os.listdir(root_path):
        category_index = file_name.split('_')[0]
        category_name = file_name.split('_')[1]
        category = category_index + '_' + category_name
        categories_dict[category] += 1
    return categories_dict


def num_object(root_path):
    categories_dict = {'1_Sand': 0, '2_Lawn': 0, '3_Bush': 0, '4_Land': 0, '5_Step': 0, '6_Mixture': 0, '7_Ground': 0,
                       '8_Playground': 0}
    for file_name in os.listdir(root_path):
        category_index = file_name.split('_')[0]
        category_name = file_name.split('_')[1]
        category = category_index + '_' + category_name
        # print file_name
        objects = parse_rbbox(os.path.join(root_path, file_name))
        categories_dict[category] += len(objects)
    return categories_dict


def rot_pts(det):
    cx, cy, h, w, angle = det[0:5]
    lt = [cx - w / 2, cy - h / 2, 1]
    rt = [cx + w / 2, cy - h / 2, 1]
    lb = [cx - w / 2, cy + h / 2, 1]
    rb = [cx + w / 2, cy + h / 2, 1]

    pts = []

    # angle = angle * 0.45

    pts.append(lt)
    pts.append(rt)
    pts.append(rb)
    pts.append(lb)

    angle = -angle

    # if angle != 0:
    cos_cita = np.cos(np.pi / 180 * angle)
    sin_cita = np.sin(np.pi / 180 * angle)

    # else :
    #	cos_cita = 1
    #	sin_cita = 0

    M0 = np.array([[1, 0, 0], [0, 1, 0], [-cx, -cy, 1]])
    M1 = np.array([[cos_cita, sin_cita, 0], [-sin_cita, cos_cita, 0], [0, 0, 1]])
    M2 = np.array([[1, 0, 0], [0, 1, 0], [cx, cy, 1]])
    rotation_matrix = M0.dot(M1).dot(M2)

    rotated_pts = np.dot(np.array(pts), rotation_matrix)

    lt = np.argmin(rotated_pts, axis=0)
    rb = np.argmax(rotated_pts, axis=0)

    left = rotated_pts[lt[0]]
    top = rotated_pts[lt[1]]
    right = rotated_pts[rb[0]]
    bottom = rotated_pts[rb[1]]

    return left, top, right, bottom


def draw_distribution(annotation_path, save_path):
    angles = []
    areas = []
    ratios = []
    for annotation in os.listdir(annotation_path):
        annotation_file = os.path.join(annotation_path, annotation)

        objects = parse_rbbox(annotation_file)

        for object_struct in objects:
            det = object_struct['bbox']
            cx, cy, w, h, angle = det

            left, top, right, bottom = rot_pts(det)

            area = w * h
            if h > w:
                ratio = h / float(w)
            else:
                ratio = w / float(h)
            dis_left_top = np.sqrt((left[0] - top[0]) ** 2 + (left[1] + top[1]) ** 2)
            dis_right_top = np.sqrt((right[0] - top[0]) ** 2 + (right[1] + top[1]) ** 2)

            if dis_left_top > dis_right_top:
                angle = 180 * np.arctan((left[1] - top[1]) / (left[0] - top[0])) / np.pi
            elif dis_left_top < dis_right_top:
                angle = 180 * np.arctan((top[1] - right[1]) / (top[0] - right[0])) / np.pi
            elif dis_left_top == dis_right_top:
                angle = 180 * np.arctan((top[1] - right[1]) / (top[0] - right[0])) / np.pi

            # print 'angle: ', angle
            if angle < 0:
                angle = 180 + angle
            angles.append((angle))
            areas.append((area))
            ratios.append(ratio)
    np_angles = np.array(angles)
    np_areas = np.array(areas)
    np_ratios = np.array(ratios)
    # hist, bins = np.histogram(np_angles, bins = np.arange(0, 360, 10))
    fig1 = plt.figure(1)
    plt.hist(np_areas, bins=np.arange(np.min(np_areas), 4000, (4000 - np.min(np_areas)) / 30), histtype='bar',
             facecolor='dodgerblue', alpha=0.75, rwidth=0.9)
    plt.title('Size distribution')
    plt.xlabel('size')
    fig1.savefig(os.path.join(save_path, 'size_hist.pdf'), bbox_inches='tight')

    fig2 = plt.figure(2)
    plt.hist(np_angles, bins=np.arange(0, 180, 5), histtype='bar', facecolor='dodgerblue', alpha=0.75, rwidth=0.9)
    plt.title('Angle distribution')
    plt.xlabel('angle')
    fig2.savefig(os.path.join(save_path, 'angle_hist.pdf'), bbox_inches='tight')

    fig3 = plt.figure(3)
    plt.hist(np_ratios, bins=np.arange(1, 5, 4 / 30.0), histtype='bar', facecolor='dodgerblue', alpha=0.75, rwidth=0.9)
    plt.title('Ratio distribution')
    plt.xlabel('ratio')
    fig3.savefig(os.path.join(save_path, 'ratio_hist.pdf'), bbox_inches='tight')

    plt.show()


def select_image_size(annotation_path, image_path, delete_file=False):
    for annotation in os.listdir(annotation_path):
        annotation_file = os.path.join(annotation_path, annotation)
        image_file = os.path.join(image_path, annotation.split('.')[0] + '.jpg')
        objects = parse_rbbox(annotation_file)
        object_struct = objects[0]
        image_size = object_struct['size']
        if image_size[0] != 342 or image_size[1] != 342:
            print
            annotation.split('.')[0]
            if delete_file == True:
                os.remove(image_file)
                os.remove(annotation_file)


def coordinate_to_xy(left, top, right, bottom, width=342, height=342):
    xmin = min(left[0], top[0], right[0], bottom[0]) + 1
    if xmin < 0:
        xmin = 1
    xmax = max(left[0], top[0], right[0], bottom[0]) - 1
    if xmax > width:
        xmax = width
    ymin = min(left[1], top[1], right[1], bottom[1]) + 1
    if ymin < 0:
        ymin = 1
    ymax = max(left[1], top[1], right[1], bottom[1]) - 1
    if ymax > height:
        ymax = height
    w = xmax - xmin
    h = ymax - ymin
    return xmin, xmax, ymin, ymax, w, h


def write_xml(structs, image_filename, image_path, save_path, object_name='bottle'):
    struct = structs[0]
    doc = Document()  # 创建DOM文档对象
    annotation = doc.createElement('annotation')
    annotation.setAttribute('verified', "no")
    doc.appendChild(annotation)

    # folder
    folder = doc.createElement('folder')
    folder_text = doc.createTextNode('JPEGImages')
    annotation.appendChild(folder)
    folder.appendChild(folder_text)

    # filename
    filename = doc.createElement('filename')
    filename_text = doc.createTextNode(image_filename)
    annotation.appendChild(filename)
    filename.appendChild(filename_text)

    # path
    path = doc.createElement('path')
    path_text = doc.createTextNode(image_path)
    annotation.appendChild(path)
    path.appendChild(path_text)

    # source
    source = doc.createElement('source')
    database = doc.createElement('database')
    database_text = doc.createTextNode('UAV-BD')
    annotation.appendChild(source)
    source.appendChild(database)
    database.appendChild(database_text)

    # size
    size = doc.createElement('size')
    width = doc.createElement('width')
    height = doc.createElement('height')
    depth = doc.createElement('depth')

    [struct_width, struct_height, struct_depth] = struct['size']
    width_text = doc.createTextNode(str(struct_width))
    height_text = doc.createTextNode(str(struct_height))
    depth_text = doc.createTextNode(str(struct_depth))

    annotation.appendChild(size)
    size.appendChild(width)
    size.appendChild(height)
    size.appendChild(depth)
    width.appendChild(width_text)
    height.appendChild(height_text)
    depth.appendChild(depth_text)

    # segmented
    segmented = doc.createElement('segmented')
    segmented_text = doc.createTextNode('0')
    annotation.appendChild(segmented)
    segmented.appendChild(segmented_text)

    # object
    for struct in structs:
        det = struct['bbox']

        left, top, right, bottom = rot_pts(det)
        struct_xmin, struct_xmax, struct_ymin, struct_ymax, struct_w, struct_h = coordinate_to_xy(left, top, right,
                                                                                                  bottom,
                                                                                                  width=struct_width,
                                                                                                  height=struct_height)

        object = doc.createElement('object')
        name = doc.createElement('name')
        pose = doc.createElement('pose')
        truncated = doc.createElement('truncated')
        difficult = doc.createElement('difficult')
        bndbox = doc.createElement('bndbox')
        xmin = doc.createElement('xmin')
        ymin = doc.createElement('ymin')
        xmax = doc.createElement('xmax')
        ymax = doc.createElement('ymax')
        name_text = doc.createTextNode(object_name)
        # print struct['name']
        pose_text = doc.createTextNode(struct['pose'])
        truncated_text = doc.createTextNode(str(struct['truncated']))
        difficult_text = doc.createTextNode(str(struct['difficult']))
        bndbox_text = doc.createTextNode('bndbox')
        xmin_text = doc.createTextNode(str(int(struct_xmin)))
        ymin_text = doc.createTextNode(str(int(struct_ymin)))
        xmax_text = doc.createTextNode(str(int(struct_xmax)))
        ymax_text = doc.createTextNode(str(int(struct_ymax)))
        annotation.appendChild(object)
        object.appendChild(name)
        object.appendChild(pose)
        object.appendChild(truncated)
        object.appendChild(difficult)
        object.appendChild(bndbox)
        bndbox.appendChild(xmin)
        bndbox.appendChild(ymin)
        bndbox.appendChild(xmax)
        bndbox.appendChild(ymax)

        name.appendChild(name_text)
        pose.appendChild(pose_text)
        truncated.appendChild(truncated_text)
        difficult.appendChild(difficult_text)
        xmin.appendChild(xmin_text)
        ymin.appendChild(ymin_text)
        xmax.appendChild(xmax_text)
        ymax.appendChild(ymax_text)

    save_file = os.path.join(save_path, image_filename.split('.')[0] + '.xml')
    print
    save_file
    f = open(save_file, 'w')
    doc.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
    f.close()


def parse_rbbox_direct(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    root = tree.getroot()
    objects = []
    image_size = root.find('size')
    width = int(image_size.find('width').text)
    height = int(image_size.find('height').text)
    depth = int(image_size.find('depth').text)
    segmented = int(root.find('segmented').text)
    for object in root.findall('object'):
        obj_struct = {}
        obj_struct['size'] = [width, height, depth]
        obj_struct['segmented'] = segmented
        obj_struct['name'] = object.find('name').text
        obj_struct['pose'] = object.find('pose').text
        obj_struct['truncated'] = int(object.find('truncated').text)
        obj_struct['difficult'] = int(object.find('difficult').text)
        robndbox = object.find('robndbox')
        cx = int(float(robndbox.find('cx').text))
        cy = int(float(robndbox.find('cy').text))
        w = int(float(robndbox.find('w').text))
        h = int(float(robndbox.find('h').text))
        angle = float(robndbox.find('angle').text)
        obj_struct['bbox'] = [cx, cy, h, w, angle]
        objects.append(obj_struct)
    return objects

def write_xml_direct_UAV_BD(structs, image_filename, image_path, save_path, object_name='bottle'):
    struct = structs[0]
    doc = Document()  # 创建DOM文档对象
    annotation = doc.createElement('annotation')
    annotation.setAttribute('verified', "no")
    doc.appendChild(annotation)

    # folder
    folder = doc.createElement('folder')
    folder_text = doc.createTextNode('JPEGImages')
    annotation.appendChild(folder)
    folder.appendChild(folder_text)

    # filename
    filename = doc.createElement('filename')
    filename_text = doc.createTextNode(image_filename)
    annotation.appendChild(filename)
    filename.appendChild(filename_text)

    # path
    path = doc.createElement('path')
    path_text = doc.createTextNode(image_path)
    annotation.appendChild(path)
    path.appendChild(path_text)

    # source
    source = doc.createElement('source')
    database = doc.createElement('database')
    database_text = doc.createTextNode('UAV-BD')
    annotation.appendChild(source)
    source.appendChild(database)
    database.appendChild(database_text)

    # size
    size = doc.createElement('size')
    width = doc.createElement('width')
    height = doc.createElement('height')
    depth = doc.createElement('depth')

    [struct_width, struct_height, struct_depth] = struct['size']
    width_text = doc.createTextNode(str(struct_width))
    height_text = doc.createTextNode(str(struct_height))
    depth_text = doc.createTextNode(str(struct_depth))

    annotation.appendChild(size)
    size.appendChild(width)
    size.appendChild(height)
    size.appendChild(depth)
    width.appendChild(width_text)
    height.appendChild(height_text)
    depth.appendChild(depth_text)

    # segmented
    segmented = doc.createElement('segmented')
    segmented_text = doc.createTextNode('0')
    annotation.appendChild(segmented)
    segmented.appendChild(segmented_text)

    # object
    for struct in structs:
        # obj_struct['bbox'] = [cx, cy, h, w, angle]
        cx_int, cy_int, h_int, w_int, angle_int = struct['bbox']

        object = doc.createElement('object')
        type = doc.createElement('type')
        name = doc.createElement('name')
        pose = doc.createElement('pose')
        truncated = doc.createElement('truncated')
        difficult = doc.createElement('difficult')
        robndbox = doc.createElement('robndbox')
        cx = doc.createElement('cx')
        cy = doc.createElement('cy')
        w = doc.createElement('w')
        h = doc.createElement('h')
        angle = doc.createElement('angle')

        type_text = doc.createTextNode('robndbox')
        name_text = doc.createTextNode(object_name)
        # print struct['name']
        pose_text = doc.createTextNode(struct['pose'])
        truncated_text = doc.createTextNode(str(struct['truncated']))
        difficult_text = doc.createTextNode(str(struct['difficult']))
        robndbox_text = doc.createTextNode('robndbox')
        cx_text = doc.createTextNode(str(float(cx_int)))
        cy_text = doc.createTextNode(str(float(cy_int)))
        w_text = doc.createTextNode(str(float(w_int)))
        h_text = doc.createTextNode(str(float(h_int)))
        angle_text = doc.createTextNode(str(float(angle_int)))

        annotation.appendChild(object)
        object.appendChild(type)
        object.appendChild(name)
        object.appendChild(pose)
        object.appendChild(truncated)
        object.appendChild(difficult)
        object.appendChild(robndbox)
        robndbox.appendChild(cx)
        robndbox.appendChild(cy)
        robndbox.appendChild(w)
        robndbox.appendChild(h)
        robndbox.appendChild(angle)

        type.appendChild(type_text)
        name.appendChild(name_text)
        pose.appendChild(pose_text)
        truncated.appendChild(truncated_text)
        difficult.appendChild(difficult_text)
        cx.appendChild(cx_text)
        cy.appendChild(cy_text)
        w.appendChild(w_text)
        h.appendChild(h_text)
        angle.appendChild(angle_text)


    save_file = os.path.join(save_path, image_filename.split('.')[0] + '.xml')
    print save_file
    f = open(save_file, 'w')
    doc.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
    f.close()



def rbbox_to_bbox(annotation_path, image_path, save_path, object_name='bottle'):
    for annotation in os.listdir(annotation_path):
        annotation_file = os.path.join(annotation_path, annotation)
        structs = parse_rbbox(annotation_file)

        image_filename = annotation.split('.')[0] + '.jpg'
        write_xml(structs, image_filename, image_path, save_path, object_name='bottle')
    print
    "finish convert!"


def generate_train_test_val(annotation_path, save_path, trainval_percentage=0.8, train_percentage=0.8):
    all_annotation = []
    for annotation_file in os.listdir(annotation_path):
        file_name = annotation_file.split('.')[0]
        all_annotation.append(file_name)
    annotation_num = len(all_annotation)
    all_annotation = np.array(all_annotation)
    np.random.shuffle(all_annotation)
    trainval_num = int(annotation_num * trainval_percentage)
    train_num = int(annotation_num * trainval_percentage * train_percentage)
    val_num = trainval_num - train_num
    test_num = annotation_num - trainval_num

    trainval_list = all_annotation[0: trainval_num - 1]
    train_list = all_annotation[0: train_num - 1]
    val_list = all_annotation[train_num - 1: trainval_num - 1]
    test_list = all_annotation[trainval_num - 1: annotation_num]

    np.savetxt(os.path.join(save_path, 'trainval.txt'), trainval_list, fmt="%s\n")
    np.savetxt(os.path.join(save_path, 'train.txt'), train_list, fmt="%s\n")
    np.savetxt(os.path.join(save_path, 'val.txt'), val_list, fmt="%s\n")
    np.savetxt(os.path.join(save_path, 'test.txt'), test_list, fmt="%s\n")


def rename_object_name_rbbox(annotation_path, object_name='bottle'):
    for annotation_file in os.listdir(annotation_path):
        annotation = os.path.join(annotation_path, annotation_file)
        tree = ET.parse(annotation)
        root = tree.getroot()

        for object in root.findall('object'):
            object_name_file = object.find('name')
            if object_name_file.text != 'bottle':
                print
                annotation_file, object_name_file.text
            object_name_file.text = object_name
        tree.write(annotation, xml_declaration=True)


def draw_box(im, BBox, color):
    cx, cy, h, w, angle = BBox[0:5]
    lt = [cx - w / 2, cy - h / 2, 1]
    rt = [cx + w / 2, cy - h / 2, 1]
    lb = [cx - w / 2, cy + h / 2, 1]
    rb = [cx + w / 2, cy + h / 2, 1]
    pts = []
    pts.append(lt)
    pts.append(rt)
    pts.append(rb)
    pts.append(lb)
    angle = -angle
    cos_cita = np.cos(np.pi / 180 * angle)
    sin_cita = np.sin(np.pi / 180 * angle)
    M0 = np.array([[1, 0, 0], [0, 1, 0], [-cx, -cy, 1]])
    M1 = np.array([[cos_cita, sin_cita, 0],
                   [-sin_cita, cos_cita, 0], [0, 0, 1]])
    M2 = np.array([[1, 0, 0], [0, 1, 0], [cx, cy, 1]])
    rotation_matrix = M0.dot(M1).dot(M2)
    rotated_pts = np.dot(np.array(pts), rotation_matrix)

    cv2.line(im, (int(rotated_pts[0, 0]), int(rotated_pts[0, 1])),
             (int(rotated_pts[1, 0]), int(rotated_pts[1, 1])), color, 5)
    cv2.line(im, (int(rotated_pts[1, 0]), int(rotated_pts[1, 1])),
             (int(rotated_pts[2, 0]), int(rotated_pts[2, 1])), color, 5)
    cv2.line(im, (int(rotated_pts[2, 0]), int(rotated_pts[2, 1])),
             (int(rotated_pts[3, 0]), int(rotated_pts[3, 1])), color, 5)
    cv2.line(im, (int(rotated_pts[3, 0]), int(rotated_pts[3, 1])),
             (int(rotated_pts[0, 0]), int(rotated_pts[0, 1])), color, 5)
    return im


def preview_annotated_image(root_path, image_path, save_path=None, bbox='bbox', display=False):
    if bbox == 'bbox':
        annotation_path = os.path.join(root_path, 'Annotations_bbox')
    if bbox == 'rbbox':
        annotation_path = os.path.join(root_path, 'Annotations_rbbox')

    for annotation in os.listdir(annotation_path):
        annotation_file = os.path.join(annotation_path, annotation)
        image_file = os.path.join(image_path, annotation.split('.')[0] + '.jpg')
        img = cv2.imread(image_file)
        if bbox == 'bbox':
            structs = parse_bbox(annotation_file)
        if bbox == 'rbbox':
            structs = parse_rbbox(annotation_file)

        for struct in structs:
            if bbox == 'bbox':
                [xmin, ymin, xmax, ymax] = struct['bbox']
                img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            if bbox == 'rbbox':
                img = draw_box(img, struct['bbox'], (0, 255, 0))

        if save_path:
            save_file = os.path.join(save_path, annotation.split('.')[0] + '.jpg')
            cv2.imwrite(save_file, img)
        if display:
            cv2.imshow('preview', img)
            cv2.waitKey(0)


def draw_PR(PR_files, file_name, color):
    fig = plt.figure()
    for algorithm, file in PR_files:
        color_algorithm = color[algorithm]
        f = open(file)
        info = pickle.load(f)
        x = info['rec']
        y = info['prec']
        print
        algorithm, info['ap']
        plt.plot(x, y, label=algorithm + ' ' + '(' + 'AP = ' + str(round(info['ap'], 3)) + ')', color=color_algorithm)

    plt.legend()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    fig.savefig(file_name, bbox_inches='tight')
    plt.show()


def open_pkl(file_name):
    f = open(file_name)
    info = pickle.load(f)
    print
    info
    # txt_file = open('sampleList.txt', 'w')
    # for detail in info:
    #     txt_file.write(str(detail))
    #     txt_file.write('\n')
    # txt_file.close()


def create_train_data(trainval_file, image_path, save_path):
    image_list = []
    with open(trainval_file, 'r') as f:
        line = f.readline()
        while line:
            line = line.strip('\n')
            line = line + '.jpg'
            image = os.path.join(image_path, line)
            print
            image
            shutil.copy(image, save_path)
            line = f.readline()


def create_trainval_txt_file(train_data_path, save_path):
    lines = []
    for image_name in os.listdir(train_data_path):
        if image_name.split('.')[1] == 'jpg':
            annotation_name = image_name + '.rbox'
            print
            image_name, annotation_name
            lines.append(image_name + ' ' + annotation_name)

    trainval_file = open(os.path.join(save_path, 'trainval.txt'), 'w')
    for line in lines:
        trainval_file.write(line)
        trainval_file.write('\n')
    trainval_file.close()


def DRBox_parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    root = tree.getroot()
    objects = []

    for object in root.findall('object'):
        obj_struct = {}
        robndbox = object.find('robndbox')
        cx = robndbox.find('cx').text
        cy = robndbox.find('cy').text
        w = robndbox.find('w').text
        h = robndbox.find('h').text
        angle = robndbox.find('angle').text
        angle = float(angle) * 180.0 / np.pi
        angle = 360 - angle
        angle = str(angle)
        obj_struct['bbox'] = cx + ' ' + cy + ' ' + h + ' ' + w + ' ' + '1' + ' ' + angle
        objects.append(obj_struct)
    return objects


def create_train_annotation_files(train_data_path, annotation_path, save_path):
    for img in os.listdir(train_data_path):
        if img.split(".")[1] == 'jpg':
            img_name = img.split(".")[0]
            objects = DRBox_parse_rec(os.path.join(annotation_path, img_name + '.xml'))
            save_name = img_name + '.jpg.rbox'
            save_file = open(os.path.join(save_path, save_name), 'w')
            for object in objects:
                box = object['bbox']
                save_file.write(box + '\n')


def preview_DRBox_annotations(root_path):
    for annotation_list in os.listdir(root_path):
        if len(annotation_list.split('.')) == 3:
            image_file = os.path.join(root_path, annotation_list.split('.')[0] + '.jpg')
            im = cv2.imread(image_file)
            with open(os.path.join(root_path, annotation_list), 'r') as f:
                line = f.readline()
                while line:
                    line = line.strip("\n")
                    line = line.split(' ')
                    line = map(eval, line)
                    box = np.array(line)
                    box = np.delete(box, 4)
                    draw_box(im, box, (0, 0, 255))
                    line = f.readline()
            cv2.imshow('preview', im)
            cv2.waitKey(0)
        else:
            continue


def num_object_imageset(imagesets_path, annotation_path):
    imagesets_dict = {'trainval': 0, 'train': 0, 'test': 0, 'val': 0}

    for file_name in os.listdir(imagesets_path):
        imagesets_name = file_name.split('.')[0]
        with open(os.path.join(imagesets_path, file_name), 'r') as f:
            line = f.readline()
            while line:
                line = line.strip('\n')
                line = line + '.xml'
                annotation = os.path.join(annotation_path, line)
                objects = parse_bbox(annotation)
                imagesets_dict[imagesets_name] += len(objects)
                line = f.readline()
    return imagesets_dict


def coordinate_to_xy_no_boundary(left, top, right, bottom):
    xmin = min(left[0], top[0], right[0], bottom[0]) + 1
    xmax = max(left[0], top[0], right[0], bottom[0]) - 1
    ymin = min(left[1], top[1], right[1], bottom[1]) + 1
    ymax = max(left[1], top[1], right[1], bottom[1]) - 1
    return xmin, ymin, xmax, ymax


def rot_pts_no_order(det):
    cx, cy, h, w, angle = det[0:5]
    lt = [cx - w / 2, cy - h / 2, 1]
    rt = [cx + w / 2, cy - h / 2, 1]
    lb = [cx - w / 2, cy + h / 2, 1]
    rb = [cx + w / 2, cy + h / 2, 1]

    pts = []

    # angle = angle * 0.45

    pts.append(lt)
    pts.append(rt)
    pts.append(rb)
    pts.append(lb)

    angle = -angle

    # if angle != 0:
    cos_cita = np.cos(np.pi / 180 * angle)
    sin_cita = np.sin(np.pi / 180 * angle)

    # else :
    #	cos_cita = 1
    #	sin_cita = 0

    M0 = np.array([[1, 0, 0], [0, 1, 0], [-cx, -cy, 1]])
    M1 = np.array([[cos_cita, sin_cita, 0], [-sin_cita, cos_cita, 0], [0, 0, 1]])
    M2 = np.array([[1, 0, 0], [0, 1, 0], [cx, cy, 1]])
    rotation_matrix = M0.dot(M1).dot(M2)

    rotated_pts = np.dot(np.array(pts), rotation_matrix)

    lt = np.argmin(rotated_pts, axis=0)
    rb = np.argmax(rotated_pts, axis=0)

    left = rotated_pts[0]
    top = rotated_pts[1]
    right = rotated_pts[2]
    bottom = rotated_pts[3]

    return left, top, right, bottom


# def xy_rotation(left, top, right, bottom, det):
#     cx, cy, h, w, angle = det
#     angle = angle * np.pi / 180.0
#     # angle 是弧度制
#     R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
#     R = np.mat(R)
#     R_I = R.I
#
#     # print left, top, right, bottom
#     print left
#     print top
#     print right
#     print bottom
#     left = R_I * np.mat(np.array([left[0] - cx, left[1] - cy]).reshape((2,1))) + np.mat(np.array([cx, cy])).reshape((2,1))
#     top = R_I * np.mat(np.array([top[0] - cx, top[1] - cy]).reshape((2, 1))) + np.mat(np.array([cx, cy])).reshape((2, 1))
#     right = R_I * np.mat(np.array([right[0] - cx, right[1] - cy]).reshape((2, 1))) + np.mat(np.array([cx, cy])).reshape((2, 1))
#     bottom = R_I * np.mat(np.array([bottom[0] - cx, bottom[1] - cy]).reshape((2, 1))) + np.mat(np.array([cx, cy])).reshape((2, 1))
#     print left
#     print top
#     print right
#     print bottom
#     return left, top, right, bottom

def xy_rotation(det):
    det = np.squeeze(det)
    cx, cy, h, w, angle = det[0:5]
    left = [cx - w / 2, cy - h / 2]
    top = [cx + w / 2, cy - h / 2]
    right = [cx - w / 2, cy + h / 2]
    bottom = [cx + w / 2, cy + h / 2]

    xmin = min(left[0], top[0], right[0], bottom[0]) + 1
    xmax = max(left[0], top[0], right[0], bottom[0]) - 1
    ymin = min(left[1], top[1], right[1], bottom[1]) + 1
    ymax = max(left[1], top[1], right[1], bottom[1]) - 1

    return [xmin, ymin, xmax, ymax]


def bbox_overlaps(bb, BBGT):
    ixmin = np.maximum(BBGT[:, 0], bb[0])
    iymin = np.maximum(BBGT[:, 1], bb[1])
    ixmax = np.minimum(BBGT[:, 2], bb[2])
    iymax = np.minimum(BBGT[:, 3], bb[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
           (BBGT[:, 2] - BBGT[:, 0] + 1.) *
           (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

    overlaps = inters / uni
    return overlaps


def attenuation_coefficient(bb, BBGT):
    angle_bb = bb[:, -1]
    angle_BBGT = BBGT[:, -1]
    angle = angle_BBGT - angle_bb
    return np.abs(np.cos(angle / 180 * np.pi))


def ArIoU(bb, BBGT):
    # 输入的 bb 为一个 rbbox, BBGT 为若干 rbbox
    # 0. 计算 “面积衰减系数”
    coefficient = attenuation_coefficient(bb, BBGT)

    # 1. 将 bb 转换成 (xmin, ymin, xmax, ymax)
    det = bb[:, 0:5]
    bb = xy_rotation(det)

    # 2. 将 BBGT 转换成若干(xmin, ymin, xmax, ymax)
    _BBGT_ = []  # 临时变量
    for idx in range(BBGT.shape[0]):
        det = BBGT[idx, :]
        _BBGT = xy_rotation(det)
        _BBGT_.append(_BBGT)

    BBGT = np.array(_BBGT_)

    # 3. 计算 bbox 的重叠面积
    overlaps = bbox_overlaps(bb, BBGT)

    # 4. 计算衰减后的面积
    overlaps = overlaps * coefficient
    return overlaps


def parse_bbox_UAV_PP(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    root = tree.getroot()
    objects = []

    for object in root.findall('object'):
        obj_struct = {}
        obj_struct['name'] = object.find('name').text
        obj_struct['pose'] = object.find('pose').text
        obj_struct['truncated'] = int(object.find('truncated').text)
        obj_struct['difficult'] = int(object.find('difficult').text)
        bndbox = object.find('bndbox')
        xmin = bndbox.find('xmin').text
        ymin = bndbox.find('ymin').text
        xmax = bndbox.find('xmax').text
        ymax = bndbox.find('ymax').text

        obj_struct['bbox'] = xmin + ' ' + ymin + ' ' + xmax + ' ' + ymax + ' '
        objects.append(obj_struct)
    return objects


def create_train_annotation_files_normal(annotation_path, save_path):
    save_name = 'UAV_PP_Annotation.txt'
    save_file = open(os.path.join(save_path, save_name), 'w')
    k = 0
    all_number = 0
    for anno_file in os.listdir(annotation_path):
        objects = parse_bbox_UAV_PP(os.path.join(annotation_path, anno_file))
        image_name = anno_file.split('.')[0] + '.jpg'
        write_detail = 'image_name'
        object_num = 0
        box = ' '
        for object in objects:
            box += object['bbox']
            object_num += 1
            all_number += 1
        save_file.write(image_name + ' ' + str(object_num) + box + '\n')
    print
    all_number


def xy_rotation_direct(det, width, height):
    det = np.squeeze(det)
    cx, cy, h, w, angle = det[0:5]
    left = [cx - w / 2, cy - h / 2]
    top = [cx + w / 2, cy - h / 2]
    right = [cx - w / 2, cy + h / 2]
    bottom = [cx + w / 2, cy + h / 2]

    xmin = min(left[0], top[0], right[0], bottom[0]) + 1
    if xmin < 0:
        xmin = 1

    xmax = max(left[0], top[0], right[0], bottom[0]) - 1
    if xmax > width:
        xmax = width

    ymin = min(left[1], top[1], right[1], bottom[1]) + 1
    if ymin < 0:
        ymin = 1

    ymax = max(left[1], top[1], right[1], bottom[1]) - 1
    if ymax > height:
        ymax = height

    return [xmin, ymin, xmax, ymax]


def write_xml_direct(structs, image_filename, image_path, save_path, object_name='bottle'):
    struct = structs[0]
    doc = Document()  # 创建DOM文档对象
    annotation = doc.createElement('annotation')
    annotation.setAttribute('verified', "no")
    doc.appendChild(annotation)

    # folder
    folder = doc.createElement('folder')
    folder_text = doc.createTextNode('JPEGImages')
    annotation.appendChild(folder)
    folder.appendChild(folder_text)

    # filename
    filename = doc.createElement('filename')
    filename_text = doc.createTextNode(image_filename)
    annotation.appendChild(filename)
    filename.appendChild(filename_text)

    # path
    path = doc.createElement('path')
    path_text = doc.createTextNode(image_path)
    annotation.appendChild(path)
    path.appendChild(path_text)

    # source
    source = doc.createElement('source')
    database = doc.createElement('database')
    database_text = doc.createTextNode('UAV-BD')
    annotation.appendChild(source)
    source.appendChild(database)
    database.appendChild(database_text)

    # size
    size = doc.createElement('size')
    width = doc.createElement('width')
    height = doc.createElement('height')
    depth = doc.createElement('depth')

    [struct_width, struct_height, struct_depth] = struct['size']
    width_text = doc.createTextNode(str(struct_width))
    height_text = doc.createTextNode(str(struct_height))
    depth_text = doc.createTextNode(str(struct_depth))

    annotation.appendChild(size)
    size.appendChild(width)
    size.appendChild(height)
    size.appendChild(depth)
    width.appendChild(width_text)
    height.appendChild(height_text)
    depth.appendChild(depth_text)

    # segmented
    segmented = doc.createElement('segmented')
    segmented_text = doc.createTextNode('0')
    annotation.appendChild(segmented)
    segmented.appendChild(segmented_text)

    # object
    for struct in structs:
        det = struct['bbox']
        struct_xmin, struct_ymin, struct_xmax, struct_ymax = xy_rotation_direct(det,
                                                                                struct_width,
                                                                                struct_height)
        # left, top, right, bottom = rot_pts(det)
        # struct_xmin, struct_xmax, struct_ymin, struct_ymax, struct_w, struct_h = coordinate_to_xy(left, top, right, bottom, width=struct_width, height=struct_height)

        object = doc.createElement('object')
        name = doc.createElement('name')
        pose = doc.createElement('pose')
        truncated = doc.createElement('truncated')
        difficult = doc.createElement('difficult')
        bndbox = doc.createElement('bndbox')
        xmin = doc.createElement('xmin')
        ymin = doc.createElement('ymin')
        xmax = doc.createElement('xmax')
        ymax = doc.createElement('ymax')
        name_text = doc.createTextNode(object_name)
        # print struct['name']
        pose_text = doc.createTextNode(struct['pose'])
        truncated_text = doc.createTextNode(str(struct['truncated']))
        difficult_text = doc.createTextNode(str(struct['difficult']))
        bndbox_text = doc.createTextNode('bndbox')
        xmin_text = doc.createTextNode(str(int(struct_xmin)))
        ymin_text = doc.createTextNode(str(int(struct_ymin)))
        xmax_text = doc.createTextNode(str(int(struct_xmax)))
        ymax_text = doc.createTextNode(str(int(struct_ymax)))
        annotation.appendChild(object)
        object.appendChild(name)
        object.appendChild(pose)
        object.appendChild(truncated)
        object.appendChild(difficult)
        object.appendChild(bndbox)
        bndbox.appendChild(xmin)
        bndbox.appendChild(ymin)
        bndbox.appendChild(xmax)
        bndbox.appendChild(ymax)

        name.appendChild(name_text)
        pose.appendChild(pose_text)
        truncated.appendChild(truncated_text)
        difficult.appendChild(difficult_text)
        xmin.appendChild(xmin_text)
        ymin.appendChild(ymin_text)
        xmax.appendChild(xmax_text)
        ymax.appendChild(ymax_text)

    save_file = os.path.join(save_path, image_filename.split('.')[0] + '.xml')
    print
    save_file
    f = open(save_file, 'w')
    doc.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
    f.close()


def rbbox_to_bbox_direct(annotation_path, image_path, save_path, object_name='bottle'):
    for annotation in os.listdir(annotation_path):
        annotation_file = os.path.join(annotation_path, annotation)
        structs = parse_rbbox(annotation_file)

        image_filename = annotation.split('.')[0] + '.jpg'
        write_xml_direct(structs, image_filename, image_path, save_path, object_name='bottle')
    print
    "finish convert!"


def xy_to_theta(BB):
    cx = (BB[:, 0] + BB[:, 2]) / 2
    cy = (BB[:, 1] + BB[:, 3]) / 2
    h = BB[:, 3] - BB[:, 1]
    w = BB[:, 2] - BB[:, 0]
    angle = np.zeros((cx.shape))
    return np.column_stack(
        (cx.reshape(-1, 1), cy.reshape(-1, 1), h.reshape(-1, 1), w.reshape(-1, 1), angle.reshape(-1, 1)))


import codecs


def xy_to_theta_result(result_file, save_path):
    # save_file = codecs.open(os.path.join(save_path, 'convert.txt'), 'w', encoding='utf-8')
    save_file = open(os.path.join(save_path, 'convert.txt'), 'w')
    with open(result_file, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
    BB = xy_to_theta(BB)
    all = np.column_stack((np.array(image_ids).reshape(-1, 1), np.array(confidence).reshape(-1, 1), BB))
    print
    all
    for i in range(all.shape[0]):
        detail = ''
        for j in range(all.shape[1]):
            detail = detail + all[i, j] + ' '
        detail.strip()
        print
        detail
        save_file.write(detail + '\n')

        # np.savetxt(os.path.join(save_path, 'convert.txt'), all)
        # save_file.write(all)
        # save_file.close()


def extract_samples_pos(save_path, annotation_path, image_path):
    pos_idx = 0
    for annotation in os.listdir(annotation_path):
        annotation_file = os.path.join(annotation_path, annotation)
        structs = parse_bbox(annotation_file)
        image_file = os.path.join(image_path, annotation.split('.')[0] + '.jpg')
        im = cv2.imread(image_file)
        pos_path = os.path.join(save_path, 'pos')
        for object in structs:
            pos_idx += 1
            bbox = object['bbox']
            im_crop = im[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
            im_crop_file = os.path.join(pos_path, 'Pos_%06d.jpg' % pos_idx)
            print
            "Saving:", pos_idx
            cv2.imwrite(im_crop_file, im_crop)


def extract_samples_neg(save_path, root_path, number_each_image):
    categories = ['1_Sand', '2_Lawn', '3_Bush', '4_Land', '5_Step', '6_Mixture', '7_Ground', '8_Playground']
    neg_idx = 22767
    crop_size = 100
    for categroy in categories:
        image_path = os.path.join(root_path, categroy)
        for image in os.listdir(image_path):
            if image.split('.')[0] == 'Cut':
                continue
            image_file = os.path.join(image_path, image)
            im = cv2.imread(image_file)
            neg_path = os.path.join(save_path, 'neg')
            for _ in range(number_each_image):
                neg_idx += 1
                bbox_x = np.random.randint(low=1, high=5471 - crop_size, size=1)
                bbox_y = np.random.randint(low=1, high=3077 - crop_size, size=1)
                im_crop = im[bbox_y[0]: bbox_y[0] + crop_size, bbox_x[0]: bbox_x[0] + crop_size, :]
                im_crop_file = os.path.join(neg_path, 'Neg_%06d.jpg' % neg_idx)
                print
                "Saving:", neg_idx
                cv2.imwrite(im_crop_file, im_crop)


def rename_samples(image_path):
    idx = 0
    for image_name in os.listdir(image_path):
        idx += 1
        image = os.path.join(image_path, image_name)
        new_image = os.path.join(image_path, "Neg_%06d.jpg" % idx)
        os.rename(image, new_image)


def mat_to_xml(structs, image_filename, image_path, save_path, object_name='bottle'):
    struct = structs[0]
    doc = Document()  # 创建DOM文档对象
    annotation = doc.createElement('annotation')
    annotation.setAttribute('verified', "no")
    doc.appendChild(annotation)

    # folder
    folder = doc.createElement('folder')
    folder_text = doc.createTextNode('JPEGImages')
    annotation.appendChild(folder)
    folder.appendChild(folder_text)

    # filename
    filename = doc.createElement('filename')
    filename_text = doc.createTextNode(image_filename)
    annotation.appendChild(filename)
    filename.appendChild(filename_text)

    # path
    path = doc.createElement('path')
    path_text = doc.createTextNode(image_path)
    annotation.appendChild(path)
    path.appendChild(path_text)

    # source
    source = doc.createElement('source')
    database = doc.createElement('database')
    database_text = doc.createTextNode('UAV-BD')
    annotation.appendChild(source)
    source.appendChild(database)
    database.appendChild(database_text)

    # size
    size = doc.createElement('size')
    width = doc.createElement('width')
    height = doc.createElement('height')
    depth = doc.createElement('depth')

    [struct_width, struct_height, struct_depth] = struct['size']
    width_text = doc.createTextNode(str(struct_width))
    height_text = doc.createTextNode(str(struct_height))
    depth_text = doc.createTextNode(str(struct_depth))

    annotation.appendChild(size)
    size.appendChild(width)
    size.appendChild(height)
    size.appendChild(depth)
    width.appendChild(width_text)
    height.appendChild(height_text)
    depth.appendChild(depth_text)

    # segmented
    segmented = doc.createElement('segmented')
    segmented_text = doc.createTextNode('0')
    annotation.appendChild(segmented)
    segmented.appendChild(segmented_text)

    # object
    for struct in structs:
        det = struct['bbox']
        struct_xmin, struct_ymin, struct_xmax, struct_ymax = det
        # left, top, right, bottom = rot_pts(det)
        # struct_xmin, struct_xmax, struct_ymin, struct_ymax, struct_w, struct_h = coordinate_to_xy(left, top, right, bottom, width=struct_width, height=struct_height)

        object = doc.createElement('object')
        name = doc.createElement('name')
        pose = doc.createElement('pose')
        truncated = doc.createElement('truncated')
        difficult = doc.createElement('difficult')
        bndbox = doc.createElement('bndbox')
        xmin = doc.createElement('xmin')
        ymin = doc.createElement('ymin')
        xmax = doc.createElement('xmax')
        ymax = doc.createElement('ymax')
        name_text = doc.createTextNode(object_name)
        # print struct['name']
        pose_text = doc.createTextNode(struct['pose'])
        truncated_text = doc.createTextNode(str(struct['truncated']))
        difficult_text = doc.createTextNode(str(struct['difficult']))
        bndbox_text = doc.createTextNode('bndbox')
        xmin_text = doc.createTextNode(str(int(struct_xmin)))
        ymin_text = doc.createTextNode(str(int(struct_ymin)))
        xmax_text = doc.createTextNode(str(int(struct_xmax)))
        ymax_text = doc.createTextNode(str(int(struct_ymax)))
        annotation.appendChild(object)
        object.appendChild(name)
        object.appendChild(pose)
        object.appendChild(truncated)
        object.appendChild(difficult)
        object.appendChild(bndbox)
        bndbox.appendChild(xmin)
        bndbox.appendChild(ymin)
        bndbox.appendChild(xmax)
        bndbox.appendChild(ymax)

        name.appendChild(name_text)
        pose.appendChild(pose_text)
        truncated.appendChild(truncated_text)
        difficult.appendChild(difficult_text)
        xmin.appendChild(xmin_text)
        ymin.appendChild(ymin_text)
        xmax.appendChild(xmax_text)
        ymax.appendChild(ymax_text)

    save_file = os.path.join(save_path, image_filename.split('.')[0] + '.xml')
    print
    save_file
    f = open(save_file, 'w')
    doc.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
    f.close()


def matlab_to_xml(mat_path, save_path, root_path, categroys):
    """
    .mat 中需要是struct结构, 不能是table
    :param mat_path:
    :param save_path:
    :return:
    """

    for categroy in categroys:
        mat_file = os.path.join(mat_path, categroy + '.mat')
        data = scio.loadmat(mat_file)
        person = data['person']
        for idx in range(person.shape[0]):
            image_struct = person[idx, 0][0]
            image_name = image_struct[0].split('\\')[-1]
            image_file = os.path.join(root_path, categroy, 'Cut', image_name)

            bbox_struct = person[idx, 0][1]

            objects = []

            width = 1000
            height = 1000
            depth = 3
            segmented = 0
            for idx_bbox in range(bbox_struct.shape[0]):
                box = bbox_struct[idx_bbox, :]
                print
                box
                obj_struct = {}
                obj_struct['size'] = [width, height, depth]
                obj_struct['segmented'] = segmented
                obj_struct['name'] = 'person'
                obj_struct['pose'] = 'Unspecified'
                obj_struct['truncated'] = 0
                obj_struct['difficult'] = 0

                xmin = box[0] + 1
                ymin = box[1] + 1
                xmax = box[0] + box[2] - 1
                ymax = box[1] + box[3] - 1

                obj_struct['bbox'] = [xmin, ymin, xmax, ymax]
                objects.append(obj_struct)

            mat_to_xml(objects, image_name, image_file, save_path, object_name='person')


def preview_annotated_image_UAV_PP(root_path, image_path, save_path=None, bbox='bbox', display=False):
    if bbox == 'bbox':
        annotation_path = os.path.join(root_path, 'Annotations_Visual')
    if bbox == 'rbbox':
        annotation_path = os.path.join(root_path, 'Annotations_rbbox')

    for annotation in os.listdir(annotation_path):
        annotation_file = os.path.join(annotation_path, annotation)
        print
        annotation_file
        image_file = os.path.join(image_path, annotation.split('.')[0] + '.jpg')
        img = cv2.imread(image_file)
        if bbox == 'bbox':
            structs = parse_bbox(annotation_file)
        if bbox == 'rbbox':
            structs = parse_rbbox(annotation_file)

        for struct in structs:
            if bbox == 'bbox':
                [xmin, ymin, xmax, ymax] = struct['bbox']
                img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            if bbox == 'rbbox':
                img = draw_box(img, struct['bbox'], (0, 255, 0))

        if save_path:
            save_file = os.path.join(save_path, annotation.split('.')[0] + '.jpg')
            cv2.imwrite(save_file, img)
        if display:
            cv2.imshow('preview', img)
            cv2.waitKey(0)


def extract_image(root_path):
    annotation_path = os.path.join(root_path, 'Annotations')
    image_path = os.path.join(root_path, 'JPEGImages')
    for annotation_file in os.listdir(annotation_path):
        objects = parse_bbox(os.path.join(annotation_path, annotation_file))
        image_file = ''
        for object in objects:
            image_file = object['path']
        print
        image_file
        shutil.copy(image_file, image_path)


def extract_Jan_image(mat_path, save_path, image_path):
    for mat in os.listdir(mat_path):
        mat_file = os.path.join(mat_path, mat)
        data = scio.loadmat(mat_file)
        person = data['person']
        for idx in range(person.shape[0]):
            image_struct = person[idx, 0][0]
            image_name = image_struct[0].split('\\')[-1]
            image_file = os.path.join(image_path, image_name)
            shutil.copy(image_file, save_path)


def rename_Jan(image_path):
    start_idx = 208
    Jan_start_idx = 444
    idx = Jan_start_idx - start_idx
    for image_name in os.listdir(image_path):
        image_file = os.path.join(image_path, image_name)
        tile_index = image_name.split('_')[3]
        Jan_idx = int(image_name.split('_')[2])
        print
        Jan_idx
        save_name = '2_Bush_' + '%06d_' % (Jan_idx - idx) + tile_index
        print
        save_name
        save_file = os.path.join(image_path, save_name)
        os.rename(image_file, save_file)


def rename_Jan_xml(annotation_path):
    start_jan_bush = 444
    start_jan_forest = 1
    start_jan_lawn = 365
    start_idx_bush = 208
    start_idx_forest = 2477
    start_idx_lawn = 133
    for anno_name in os.listdir(annotation_path):
        if anno_name.split('_')[0] != 'Jan':
            continue
        jan_idx = int(anno_name.split('_')[2])
        categroy = ''
        idx = 0
        print jan_idx
        if jan_idx >= 365 and jan_idx <= 443:  # lawn
            idx = start_jan_lawn - start_idx_lawn
            categroy = '1_Lawn'

        if jan_idx >= 1 and jan_idx <= 364:  # lawn
            idx = start_jan_forest - start_idx_forest
            categroy = '3_Forest'

        if jan_idx >= 444:  # lawn
            idx = start_jan_bush - start_idx_bush
            categroy = '2_Bush'

        tile_index = anno_name.split('_')[3]
        save_name = categroy + '_' + '%06d_' % (jan_idx - idx) + tile_index
        print save_name
        os.rename(os.path.join(annotation_path, anno_name), os.path.join(annotation_path, save_name))


def rename_xml_path(annotations_path, image_path, save_path):
    for anno_name in os.listdir(annotations_path):
        objects = parse_bbox(os.path.join(annotations_path, anno_name))
        filename = anno_name.split('.')[0] + '.jpg'
        new_path = os.path.join(image_path, anno_name.split('.')[0] + '.jpg')
        # objects['filename'] = filename
        # objects['path'] = new_path

        mat_to_xml(objects, filename, new_path, save_path, object_name='person')


def num_each_category_UAV_PP(root_path):
    categories_dict = {'1_Lawn': 0, '2_Bush': 0, '3_Forest': 0, '4_Marshland': 0, '5_Park': 0, '6_Hillside': 0,
                       '7_Grove': 0}
    for file_name in os.listdir(root_path):
        category_index = file_name.split('_')[0]
        category_name = file_name.split('_')[1]
        category = category_index + '_' + category_name
        categories_dict[category] += 1
    return categories_dict


def num_object_UAV_PP(root_path):
    categories_dict = {'1_Lawn': 0, '2_Bush': 0, '3_Forest': 0, '4_Marshland': 0, '5_Park': 0, '6_Hillside': 0,
                       '7_Grove': 0}
    for file_name in os.listdir(root_path):
        category_index = file_name.split('_')[0]
        category_name = file_name.split('_')[1]
        category = category_index + '_' + category_name
        # print file_name
        objects = parse_bbox(os.path.join(root_path, file_name))
        categories_dict[category] += len(objects)
    return categories_dict


def merge_xml(src, dst, image_path, save_path):
    for anno in os.listdir(src):
        src_annotation_file = os.path.join(src, anno)
        src_objects = parse_rbbox_direct(src_annotation_file)

        dst_annotation_file = os.path.join(dst, anno)
        dst_objects = parse_rbbox_direct(dst_annotation_file)

        objects = src_objects + dst_objects

        image_filename = anno.split('.')[0]
        save_path = save_path
        write_xml_direct_UAV_BD(objects,
                         image_filename,
                         image_path,
                         save_path,
                         object_name='bottle')




if __name__ == "__main__":
    # 1. 检查图像和标注文件是否匹配
    # check_image_annotations(delete_files=False)

    # 2. 将两阶段的图像和标注文件重命名
    # rename_two_phase_file('E:/jwwangchn/Data/UAV-Bottle/UAV-Bottle-V3.1.0/Categroies/Phase1', phase = '1')

    # 3. 统计各个类别图片数量
    # object_dict = num_each_category('E:/jwwangchn/Data/UAV-Bottle/UAV-Bottle-V3.1.0/JPEGImages')
    # print object_dict
    # print "Number of objects: ", sum(object_dict.values())

    # 4. 统计各个类别中目标数量
    # object_dict = num_object('E:/jwwangchn/Data/UAV-Bottle/UAV-Bottle-V3.1.0/Annotations')
    # print object_dict
    # print "Number of objects: ", sum(object_dict.values())

    # 5. 分布图
    # annotation_path = 'E:/jwwangchn/Data/UAV-Bottle/UAV-Bottle-V3.1.0/Annotations'
    # save_path = 'E:/jwwangchn/Software/Bottle-Detection-Paper/images/'
    # draw_distribution(annotation_path=annotation_path, save_path=save_path)

    # 6. 筛选图片尺寸不是 342*342 的图片, 并删除
    # annotation_path = 'E:/jwwangchn/Data/UAV-Bottle/UAV-Bottle-V3.1.0/Annotations'
    # image_path = 'E:/jwwangchn/Data/UAV-Bottle/UAV-Bottle-V3.1.0/JPEGImages'
    # select_image_size(annotation_path, image_path, delete_file = False)

    # 7. 生成最小外接矩形框的标注文件
    # annotation_path = '/home/ubuntu/data/VOCdevkit/UAV-BD/Annotations_rbbox'
    # save_path = '/home/ubuntu/data/VOCdevkit/UAV-BD/Annotations_bbox'
    # image_path = '/home/ubuntu/data/VOCdevkit/UAV-BD/JPEGImages'
    #
    # rbbox_to_bbox(annotation_path, image_path, save_path, object_name = 'bottle')

    # 8. 生成 trainval train test val 文件
    # annotation_path = 'E:/jwwangchn/Data/UAV-Bottle/UAV-Bottle-V3.1.0/Annotations'
    # save_path = 'E:/jwwangchn/Data/UAV-Bottle/UAV-Bottle-V3.1.0/ImageSets/Main'
    # generate_train_test_val(annotation_path, save_path, 0.8, 0.8)

    # 9. Rename object_name of rbbox
    # annotation_path = '/home/ubuntu/data/VOCdevkit/UAV-BD/Annotations_rbbox'
    # rename_object_name_rbbox(annotation_path, object_name = 'bottle')

    # 10. Preview annotation file
    root_path = './UAV-BD/UAV-Bottle-V2.0.0'
    image_path = './UAV-BD/UAV-Bottle-V2.0.0/JPEGImages'
    # save_path = 'H:/Data/UAV-Bottle/UAV-Bottle-V3.2.0/Visual'
    preview_annotated_image(root_path, image_path, save_path = None, bbox = 'rbbox', display=True)

    # 11. Draw P-R curve
    # rbbox
    # SSD_file = ('SSD', '//90.0.0.50/Documents/jwwangchn/ICIP/pr/ssd_bottle_pr_rbbox.pkl')
    # FasterRCNN_file = ('Faster R-CNN', '//90.0.0.50/Documents/jwwangchn/ICIP/pr/faster_rcnn_bottle_pr_rbbox.pkl')
    # RRPN_file = ('RRPN', '//90.0.0.50/Documents/jwwangchn/ICIP/pr/rrpn_bottle_pr_rbbox.pkl')
    # YOLOv2_file = ('YOLOv2', '//90.0.0.50/Documents/jwwangchn/ICIP/pr/yolov2_bottle_pr_rbbox.pkl')
    # # DRBox_file = ('DRBox', '//90.0.0.50/Documents/jwwangchn/ICIP/pr/drbox_bottle_pr_rbbox.pkl')
    #
    # PR_files = [RRPN_file, SSD_file, FasterRCNN_file, YOLOv2_file]
    # color = {'RRPN':'#f03b20', 'Faster R-CNN':'#2b8cbe', 'SSD':'#fec44f', 'YOLOv2':'#a1d99b'}
    # draw_PR(PR_files, 'pr_rbbox.pdf', color)
    #
    # # bbox
    # SSD_file = ('SSD', '//90.0.0.50/Documents/jwwangchn/ICIP/pr/ssd_bottle_pr_bbox.pkl')
    # FasterRCNN_file = ('Faster R-CNN', '//90.0.0.50/Documents/jwwangchn/ICIP/pr/faster_rcnn_bottle_pr_bbox.pkl')
    # YOLOv2_file = ('YOLOv2', '//90.0.0.50/Documents/jwwangchn/ICIP/pr/yolov2_bottle_pr_bbox.pkl')
    # PR_files = [FasterRCNN_file, SSD_file, YOLOv2_file]
    # color = {'RRPN': '#f03b20', 'Faster R-CNN': '#2b8cbe', 'SSD': '#fec44f', 'YOLOv2': '#a1d99b'}
    # draw_PR(PR_files, 'pr_bbox.pdf', color)

    # 12. Open pkl file
    # file_name = 'E:/jwwangchn/坚果云/文档/ubuntu/ICIP/prior_boxes.pkl'
    # file_name = unicode(file_name, 'utf8')
    # open_pkl(file_name)

    # 13. DRBox create train image data
    # trainval_file = '/home/ubuntu/data/VOCdevkit/UAV-Bottle-V3.2.0/ImageSets/Main/trainval.txt'
    # image_path = '/home/ubuntu/data/VOCdevkit/UAV-Bottle-V3.2.0/JPEGImages'
    # save_path = '/home/ubuntu/Documents/DRBox/data/bottle/train_data'
    # create_train_data(trainval_file, image_path, save_path)

    # 14. DRBox create trainval.txt file
    # train_data_path = '/home/ubuntu/Documents/DRBox/data/bottle/train_data'
    # save_path = '/home/ubuntu/Documents/DRBox/data/bottle'
    # create_trainval_txt_file(train_data_path, save_path)

    # 15. DRBox create annotation files
    # train_data_path = '/home/ubuntu/Documents/DRBox/data/bottle/train_data'
    # annotation_path = '/home/ubuntu/data/VOCdevkit/UAV-Bottle-V3.2.0/Annotations_rbbox'
    # save_path = '/home/ubuntu/Documents/DRBox/data/bottle/train_data'
    # create_train_annotation_files(train_data_path, annotation_path, save_path)

    # 16. Draw and test the annotation files
    # root_path = 'E:/jwwangchn/Data/UAV-Bottle/DRBox/DRBox_data'

    # root_path = '~/Documents/DRBox/data/bottle/train_data'
    # preview_DRBox_annotations(root_path)

    # 17. Calculate the instance number of train and test data
    # imagesets_path = 'E:/jwwangchn/Data/UAV-Bottle/UAV-Bottle-V3.2.0/ImageSets/Main'
    # annotation_path = 'E:/jwwangchn/Data/UAV-Bottle/UAV-Bottle-V3.2.0/Annotations_bbox'
    # imagesets_dict = num_object_imageset(imagesets_path, annotation_path)
    # print imagesets_dict
    # print "Number of imagesets: ", sum(imagesets_dict.values())

    # 18. 计算 ArIoU
    # bb = np.array([[100.0, 100.0, 10.0, 100.0, 0]])  # keep
    #
    # BBGT = np.array([[100.0, 100.0, 10.0, 100.0, 30],
    #                  [100.0, 100.0, 10.0, 100.0, 45],
    #                  [100.0, 100.0, 10.0, 100.0, 60],
    #                  [100.0, 100.0, 10.0, 100.0, 90]])
    # ArIoU(bb, BBGT)

    # 19. UAV-PP Dataset 传统算法转换
    # annotation_path = 'H:/Data/UAV-Bottle/UAV-Bottle-V3.2.0/Annotations_bbox'
    # save_path = 'H:/Data/UAV-Bottle/UAV-Bottle-V3.2.0/test'
    # create_train_annotation_files_normal(annotation_path, save_path)

    # 20. 不考虑角度直接生成标注文件
    # annotation_path = 'H:/Data/UAV-Bottle/UAV-Bottle-V3.2.0/Annotations_rbbox'
    # save_path = 'H:/Data/UAV-Bottle/UAV-Bottle-V3.2.0/Annotations_bbox_direct'
    # image_path = 'H:/Data/UAV-Bottle/UAV-Bottle-V3.2.0/JPEGImages'
    #
    # rbbox_to_bbox_direct(annotation_path, image_path, save_path, object_name='bottle')

    # 21. 检测结果转换
    # result_file = 'E:/result_yolo.txt'
    # save_path = 'E:/'
    # xy_to_theta_result(result_file, save_path)

    # 22. 提取正样本
    # save_path = 'H:/Data/UAV-Bottle/UAV-Bottle-V3.2.0/Samples'
    # annotation_path = 'H:/Data/UAV-Bottle/UAV-Bottle-V3.2.0/Annotations_bbox'
    # image_path = 'H:/Data/UAV-Bottle/UAV-Bottle-V3.2.0/JPEGImages'
    # extract_samples_pos(save_path, annotation_path, image_path)

    # 23. 提取负样本
    # save_path = 'H:/Data/UAV-Bottle/UAV-Bottle-V3.2.0/Samples'
    # root_path = 'H:/Data/UAV-Bottle/UAV-Bottle-V3.0.0/RAW'
    # extract_samples_neg(save_path, root_path, number_each_image = 10)

    # 24. rename neg sample
    # image_path =  'H:/Data/UAV-Bottle/UAV-Bottle-V3.2.0/Samples/Neg'
    # rename_samples(image_path)

    # 24. .mat 标注文件转换成 .xml 文件
    # mat_path = 'H:/Data/UAV-PP/UAV-PP-V2.0.0/save_mat'
    # save_path = 'H:/Data/UAV-PP/UAV-PP-V2.2.0/Annotations'
    # categroys = ['1_Lawn', '2_Bush', '3_Forest', '4_Marshland', '5_Park', '6_Hillside', '7_Grove'];
    # root_path = 'H:\Data\UAV-PP\UAV-PP-V2.0.0'
    # matlab_to_xml(mat_path, save_path, root_path, categroys)

    # 25. Preview annotation file of UAV-PP
    # root_path = 'H:/Data/UAV-PP/UAV-PP-V2.2.0'
    # image_path = 'H:/Data/UAV-PP/UAV-PP-V2.2.0/JPEGImages'
    # # save_path = 'H:/Data/UAV-PP/UAV-PP-V2.2.0/Visual'
    # preview_annotated_image_UAV_PP(root_path, image_path, bbox = 'bbox', display=True)

    # 26. UAV-PP Dataset 传统算法转换
    # annotation_path = 'H:/Data/UAV-PP/UAV-PP-V2.2.0/Annotations'
    # save_path = 'H:/Data/UAV-PP/UAV-PP-V2.2.0/test'
    # create_train_annotation_files_normal(annotation_path, save_path)

    # 27. 从标记文件提取图片
    # root_path = 'H:/Data/UAV-PP/UAV-PP-V2.2.0'
    # extract_image(root_path)

    # jan_image_path = u'H:/Data/UAV/UAV-PP-V2.0.0/RAW/1月20日/Cut'
    # mat_path = 'H:/Data/UAV-PP/UAV-PP-V2.0.0/person_all_save'
    # save_path = 'H:/Data/UAV-PP/UAV-PP-V2.0.0/Jan_forest'
    # extract_Jan_image(mat_path, save_path, jan_image_path)

    # 28. 重命名 Jan 开头的图片
    # rename_Jan('H:/Data/UAV-PP/UAV-PP-V2.0.0/Jan_bush')

    # 29. 重命名 Jan 开头的 xml 文件
    # rename_Jan_xml('H:/Data/UAV-PP/UAV-PP-V2.2.0/Annotations_test')

    # 30. 重命名 Jan xml 文件中的 path
    # image_path = 'H:/Data/UAV-PP/UAV-PP-V2.2.0/JPEGImages'
    # annotation_path = 'H:/Data/UAV-PP/UAV-PP-V2.2.0/Annotations_test'
    # save_path = 'H:/Data/UAV-PP/UAV-PP-V2.2.0/Annotations_save'
    # rename_xml_path(annotation_path, image_path, save_path)

    # 31. 预览 Jan 文件
    # root_path = 'H:/Data/UAV-PP/UAV-PP-V2.2.0'
    # image_path = 'H:/Data/UAV-PP/UAV-PP-V2.0.0/Jan_modify'
    # # save_path = 'H:/Data/UAV-PP/UAV-PP-V2.2.0/Visual'
    # preview_annotated_image_UAV_PP(root_path, image_path, bbox = 'bbox', display=True)

    # 32. 统计各个类别图片数量
    # object_dict = num_each_category_UAV_PP('F:/Data/UAV-PP/UAV-PP-V2.2.0/JPEGImages')
    # print object_dict
    # print "Number of objects: ", sum(object_dict.values())
    #
    # # 33. 统计各个类别中目标数量
    # object_dict = num_object_UAV_PP('F:/Data/UAV-PP/UAV-PP-V2.2.0/Annotations')
    # print object_dict
    # print "Number of objects: ", sum(object_dict.values())

    # 34. 预览 UAV_PP 数据集
    # root_path = 'F:/Data/UAV-PP/UAV-PP-V2.2.0'
    # image_path = 'F:/Data/UAV-PP/UAV-PP-V2.2.0/JPEGImages'
    # save_path = 'F:/Data/UAV-PP/UAV-PP-V2.2.0/Visual'
    # preview_annotated_image_UAV_PP(root_path, image_path, save_path=save_path, bbox='bbox', display=False)

    # 35. 根据 annotation 文件提取图片
    # src = './UAV-BD/补标/label'
    # # dst = './UAV-BD/UAV-Bottle-V3.2.0/Annotations_rbbox_merge'
    # dst = './UAV-BD/UAV-Bottle-V3.2.0/Annotations_rbbox_add'
    # save_path = './UAV-BD/UAV-Bottle-V3.2.0/Annotations_rbbox_add'
    # image_path = '/home/jwwangchn/data/VOCdevkit/UAV-Bottle/UAV-Bottle-V3.2.0/JPEGImages'
    # merge_xml(src, dst, image_path, save_path)
