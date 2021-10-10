import shutil
import sys
import os

def copy_paste_file(txt_file, label_dir, new_label_dir):
    if not os.path.exists(new_label_dir): 
        os.mkdir(new_label_dir)
        os.mkdir(os.path.join(new_label_dir, 'labels'))
        os.mkdir(os.path.join(new_label_dir, 'images'))

    with open(txt_file, 'r') as f:
        contents = list(map(str.strip, f.readlines()))
       
    for con in contents:
        new_path = "%s.txt"%con.split('.')[0]
        old_path = os.path.join(label_dir, new_path)
        new_path = new_path.split('/')[-1]
        new_path_label = os.path.join(new_label_dir, 'labels', new_path)
        new_path_images = os.path.join(new_label_dir, 'images', con.split('/')[-1])
        assert os.path.exists(old_path)
        shutil.copy(old_path, new_path_label)
        shutil.copy(con, new_path_images)
        print(new_path, old_path, new_path_images)
    return 0

if __name__ == '__main__':
    copy_paste_file('../voc2/2007_test.txt', '../voc2/VOC2007/JPEGImages', '../voc2/VOC2007_test')