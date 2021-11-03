import os 
import pandas as pd 


def generate_dataset_csv(path, name, save_dir):
    slides = os.listdir(path)
    res = []
    for slide in slides:
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
    suffix = 'train'
    src = "/media/ldy/e5a10f4e-18fd-4656-80d8-055bc4078655/OSCC-Tile-v2/5x_1600/"
    path = src + suffix
    save_dir = src
    name = suffix + "_1600.csv"

    generate_dataset_csv(path, name, save_dir)