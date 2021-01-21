from PIL import Image
import pdb
import numpy as np
import os
import sys

"""
Script to convert image files from AIST (sdbs.db.aist.go.jp) into a digitized absorbance.
Input in form of 'python aist_image_processing path_to_images_directory path_to_output_directory'
"""

image_directory=sys.argv[1]
output_directory=sys.argv[2]

MIN_Y, MAX_Y = 0, 100
MIN_X, MAX_X = 4000,400
MID_X=2000
highscale=100
lowscale=200
midfraction=((MID_X-MIN_X)/lowscale)/((MID_X-MIN_X)/lowscale+(MAX_X-MID_X)/highscale)

N_Y_TICKS = 2
N_X_TICKS = 2


def convert_to_numpy(im):
    dim_x, dim_y = im.size

    np_im = np.zeros([dim_x, dim_y])

    for x in range(dim_x):
        for y in range(dim_y):
            pixel = im.getpixel((x, y))
            # np_im[x,y]=pixel
            if all([rbg<100 for rbg in pixel]):
                np_im[x, y] = 1
            # elif pixel != (255, 255, 255):
                # Different color from black/white
                # assert(False)
    # print(np_im)
    return np_im


def find_graph_box(im):
    dim_x, dim_y = im.shape

    sum_pix_x = np.sum(im, axis=0)
    sum_pix_y = np.sum(im, axis=1)

    x_lines = np.where(sum_pix_x>0.58 * dim_x)[0]
    y_lines = np.where(sum_pix_y>0.58 * dim_y)[0]

    # pick the largest box for the graph:

    m_length, m_box = 0, (0, 0)
    cur_left = -1
    for x_idx in x_lines:
        if cur_left == -1:
            cur_left = x_idx
            continue
        cur_box = (cur_left, x_idx)
        box_length = x_idx - cur_left

        if box_length > m_length:
            m_length = box_length
            m_box = cur_box

        cur_left = x_idx

    if len(y_lines) > 2:
        assert(False)

    box = (m_box, (y_lines[0], y_lines[1]))
    return box


def parse_graph(graph_im):
    dim_x, dim_y = graph_im.shape

    data = []

    for x in range(dim_x):
        for y in range(dim_y):
            if graph_im[x, y] == 1:
                if x<(midfraction*dim_x):
                    label_x=(float(x)/(midfraction*dim_x)*(MID_X-MIN_X)+MIN_X)
                else:
                    label_x=((float(x)-(midfraction*dim_x))/(dim_x-midfraction*dim_x)*(MAX_X-MID_X)+MID_X)
                label_y = (dim_y - float(y)) / dim_y * MAX_Y
                data.append((label_x, label_y))
    return data


def remove_ticks(graph_im):
    dim_x, dim_y = graph_im.shape

    num_y_ticks = 0
    for y in range(dim_y):
        is_tick = True
        for i in range(N_Y_TICKS):
            if graph_im[i, y] == 0:
                is_tick = False
                break
        if is_tick:
            num_y_ticks += 1
            tick_idx = 0
            while True:
                if graph_im[tick_idx, y] == 1:
                    graph_im[tick_idx, y] = 0
                    tick_idx += 1
                else:
                    break

    num_x_ticks = 0
    for x in range(dim_x):
        is_tick = True
        for i in range(N_X_TICKS):
            if graph_im[x, -(i+1)] == 0:
                is_tick = False
                break
        if is_tick:
            num_x_ticks += 1
            tick_idx = 0
            while True:
                if graph_im[x, -(tick_idx+1)] == 1:
                    graph_im[x, -(tick_idx+1)] = 0
                    tick_idx += 1
                else:
                    break


def write_one_spectrum(file,image_directory,output_directory):
    im = Image.open(os.path.join(image_directory,file))
    rgb_im = im.convert('RGB')

    np_im = convert_to_numpy(rgb_im)

    x_box, y_box = find_graph_box(np_im)
    graph_im = np_im[y_box[0]+1:y_box[1], x_box[0]+1:x_box[1]]
    remove_ticks(graph_im)

    data = parse_graph(graph_im)
    with open(os.path.join(output_directory,file[:-4]+'.txt'),'w') as f:
        f.write('wavenumber\ttransmittance\n')
        for i in data:
            f.write(str(i[0])+'\t'+str(i[1])+'\n')

def main():
    files=os.listdir(image_directory)
    for file in files:
        # print(file)
        write_one_spectrum(file,image_directory,output_directory)


if __name__ == '__main__':
    main()
