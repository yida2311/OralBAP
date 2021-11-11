import os 
import pandas as pd 
import json


def generate_dataset_csv(slide_list, path, name, save_dir):
    # slides = os.listdir(path)
    res = []
    for slide in slide_list:
        slide_dir = os.path.join(path, slide)
        for patch in os.listdir(slide_dir):
            info = {}
            info['slide_id'] = slide 
            info['image_id'] = patch 
            row, col = parse_patch_name(patch)
            info['row'] = row 
            info['col'] = col 
            res.append(info)
    
    print("{} patches!".format(len(res)))
    df = pd.DataFrame(res, columns=['image_id', 'slide_id', 'row', 'col'])
    df.to_csv(os.path.join(save_dir, name))


def parse_patch_name(patch):
    info = patch.split('_')
    row = int(info[-3])
    col = int(info[-2])

    return row, col 


if __name__ == '__main__':
    with open('/media/ldy/7E1CA94545711AE6/OSCC/train_val_part.json') as f:
        cnt = json.load(f)
    suffix = 'val'
    src = "/media/ldy/7E1CA94545711AE6/OSCC/2.5x_tile/2.5x_640/"
    slide_list = cnt[suffix]
    path = src + 'patch'
    save_dir = src
    name = suffix + ".csv"

    generate_dataset_csv(slide_list, path, name, save_dir)