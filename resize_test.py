import numpy as np
import cv2


im = cv2.imread('/media/moon/extraDB/KetiDBconverter_local/WORKSPACE/original_dataset/kitti/testing/image_2/000000.png')
src_height = im.shape[0]
src_width = im.shape[1]

dst_height = int(src_height / 2)
dst_width = int(src_width / 2)
# dst_height = 608
# dst_width = 608
dst_im = np.zeros((dst_height, dst_width, 3), dtype='uint8')

fx = src_width / dst_width
fy = src_height / dst_height

for dst_y in range(dst_height):
    for dst_x in range(dst_width):
        src_x = (dst_x + 0.5) * fx - 0.5
        src_y = (dst_y + 0.5) * fy - 0.5

        x1 = round(src_x)
        y1 = round(src_y)
        x1_read = max(x1, 0)
        y1_read = max(y1, 0)

        x2 = x1 + 1
        y2 = y1 + 1
        x2_read = min(x2, src_width - 1)
        y2_read = min(y2, src_height - 1)

        src_reg = 0

        for c in range(3):
            out = 0.0

            src_reg = im[y1_read][x1_read][c]
            out += (x2 - src_x) * (y2 - src_y) * src_reg

            src_reg = im[y1_read][x2_read][c]
            out += (src_x - x1) * (y2 - src_y) * src_reg

            src_reg = im[y2_read][x1_read][c]
            out += (x2 - src_x) * (src_y - y1) * src_reg

            src_reg = im[y2_read][x2_read][c]
            out += (src_x - x1) * (src_y - y1) * src_reg

            if out < 0:
                out = 0
            elif out > 255:
                out = 255

            dst_im[dst_y][dst_x][c] = int(out)

im = im.reshape((src_height, src_width, 3))
cv2.imshow('src', im)
cv_resize = cv2.resize(im, (dst_width, dst_height))
cv2.imshow('cv', cv_resize)
dst_im = dst_im.reshape((dst_height, dst_width, 3))
cv2.imshow('dst', dst_im)

diff = (cv_resize / 255 - dst_im / 255) / 1
mse = np.mean(np.power(diff, 2))
print(f'MSE : {mse}')
print(f'PSNR : {-10 * np.log10(mse)}')
cv2.waitKey(0)
