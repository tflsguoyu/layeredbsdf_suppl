import cv2
import struct

input_file = "a_eval.exr"
output_file = "output.vol"

img = cv2.imread(input_file, -1)

with open(output_file, "wb") as fout:
    fout.write("VOL")
    fout.write("\x03")
    fout.write(struct.pack('I', 1))
    fout.write(struct.pack('3I', img.shape[1], img.shape[0], 1))
    fout.write(struct.pack('I', img.shape[2]))
    fout.write(struct.pack('6f', -0.5, -0.5, -0.5, 0.5, 0.5, 0.5))
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            for k in range(0, img.shape[2]):
                fout.write(struct.pack('f', img[i, j, k]))
