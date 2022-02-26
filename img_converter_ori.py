import glob
import numpy as np
import cv2
import argparse
import sys
import os
import math
'''
Example1. From Depth16 to png:
python img_converter.py -if depth16 -of png -c 640 -r 480 -i img/depth16_640x480/*.raw -o out/img/depth16_640x480/

Example2. From mipi raw to png
python img_converter.py -if mipi10 -of png -c 1280 -r 962 -i img/phase_packed_raw10_1280x962_s1600/1/*.raw -o out/img//phase_packed_raw10_1280x962_s1600/1

Example3. From UnPacked raw to png
python img_converter.py -if raw10 -of png -c 1280 -r 962 -i img/phase_unpacked_raw10_1280x962_s2560/1/*.raw -o out/img/phase_unpacked_raw10_1280x962_s2560/1

Example4. From mipi raw to bayer(multi panels)
python img_converter.py -if mipi10 -of bayer16 -c 1280 -r 962 -i img/phase_packed_raw10_1280x962_s1600/1/*.raw -o out/img/phase_packed_raw10_1280x962_s1600/1_bayer

Example5. From raw10 to png.
python img_converter.py -if raw10 -of png -c 640 -r 481 -i out/img/phase_packed_raw10_1280x962_s1600/1_bayer/*.bayer16 -o out/img/phase_packed_raw10_1280x962_s1600/1_png

Example6. From mipi raw to depth
python img_converter.py -if mipi10 -of depth16 -c 1280 -r 962 -i img/phase_packed_raw10_1280x962_s1600/1/*.raw -o out/img/phase_packed_raw10_1280x962_s1600/1_depth16

Example7. Depth16 to png
python img_converter.py -if depth16 -of png -c 640 -r 480 -i out/img/phase_packed_raw10_1280x962_s1600/1_depth16/*.depth16 -o out/img/phase_packed_raw10_1280x962_s1600/1_png

Example8 from raw10 to depth16
python img_converter.py -if raw10 -of depth16 -c 1280 -r 962 -i img/phase_unpacked_raw10_1280x962_s2560/1/*.raw -o out/img/phase_unpacked_raw10_1280x962_s2560/1_depth16

python img_converter.py -if depth16 -of png -c 640 -r 480 -i out/img/phase_unpacked_raw10_1280x962_s2560/1_depth16/*.depth16 -o out/img/phase_unpacked_raw10_1280x962_s2560/1_png

python img_converter.py -if raw10 -of depth16 -c 1280 -r 962 -i E:\\xinhai8cam\\T25N02C\\T25N02C\\*.raw -o E:\\xinhai8cam\\T25N02C\\T25N02C_DEPTH16
python img_converter.py -if depth16 -of png -c 640 -r 480 -i E:\\xinhai8cam\\T25N02C\\T25N02C_DEPTH16\\*.depth16 -o  E:\\xinhai8cam\\T25N02C\\T25N02C_DEPTH_PNG

python img_converter.py -if raw10 -of bayer16 -c 1280 -r 962 -i E:\\xinhai8cam\\T25N02C\\T25N02C\\*.raw -o E:\\xinhai8cam\\T25N02C\\T25N02C_BAYER
python img_converter.py -if raw10 -of png -c 640 -r 481 -i E:\\xinhai8cam\\T25N02C\\T25N02C_BAYER\\*.bayer16 -o E:\\xinhai8cam\\T25N02C\\T25N02C_BAYER_PNG

python img_converter.py -if raw10 -of depth16 -c 1280 -r 962 -i D:\\xinhai8cam\\1280_962\\*.raw -o D:\\xinhai8cam\\1280_962_Depth16

python img_converter.py -if raw10 -of png -c 1280 -r 962 -i D:\\xinhai8cam\\20201106\\*.raw -o D:\\xinhai8cam\\20201106_out
'''

index_embed_frame_count = 16
index_embed_phase = 182
index_embed_frequency_modulation = 184

value_embed_phase_sh = 0x1B
value_embed_phase_ns = 0x10
value_embed_frequency_modulation_100 = 0x00
value_embed_frequency_modulation_80 = 0x01
FOV_HOR = 59.2*np.pi/180.0
FOV_VER = 46.0*np.pi/180.0
depth16_img_width = 640.0
depth16_img_height = 480.0


def get_z_cal_coeff(x, y, w, h, fov_hor):
    pz = (w/2.0)/np.tan(fov_hor/2.0)
    px = x - w/2.0
    py = y - h/2.0
    l = math.sqrt(px*px + py*py + pz*pz)
    sin_theta = pz/l
    return sin_theta


def unpack_data_12(data, shape):
    result = np.zeros(shape[0] * shape[1], "uint16")
    j = 0
    for ix in range(0, len(data), 3):
        result[j] = (data[ix] << 4) | ((data[ix + 2]) >> 0 & 0xF)
        result[j + 1] = (data[ix + 1] << 4) | ((data[ix + 2] >> 4) & 0xF)
        j = j + 2
    return result


def unpack_data_10(data, shape):
    result = np.zeros(shape[0]*shape[1], "uint16")
    j = 0
    for ix in range(0, len(data), 5):
        result[j] = (data[ix] << 2) | ((data[ix + 4]) >> 0 & 0x3)
        result[j + 1] = (data[ix + 1] << 2) | ((data[ix + 4] >> 2) & 0x3)
        result[j + 2] = (data[ix + 2] << 2) | ((data[ix + 4] >> 4) & 0x3)
        result[j + 3] = (data[ix + 3] << 2) | ((data[ix + 4] >> 6) & 0x3)
        j = j + 4
    return result


def get_embeded_data(format, meta,  data, meta_file):
    if meta == 0:
        if format == "mipi10":
            return get_embeded_data_mip10(data)
        elif format == "raw10":
            return get_embeded_data_raw10(data)
    else:
        f = open(meta_file, "r+b")
        return f.read()
    return None


def get_embeded_data_mip10(data):
    emb_size = 512
    result = np.zeros(emb_size)
    j = 0
    for i in range(0, 512, 4):
        result[i] = data[j]
        result[i + 1] = data[j + 1]
        result[i + 2] = data[j + 2]
        result[i + 3] = data[j + 3]
        j = j + 5
    return result


# d: data
# b: empty bytes
#little endinan data layout
# | b15 | b14 | b13 | b12 | b11 | b10 | d7 | d6 |
# | d5  | d4  |  d3 | d2  | d1  | d0  | b1 | b0 |
#memory layout
#  | d5  | d4  |  d3 | d2  | d1  | d0  | b1 | b0 |
#  | b15 | b14 | b13 | b12 | b11 | b10 | d7 | d6 |
def get_embeded_data_raw10(data):
    emb_size = 512
    result = np.zeros(emb_size)
    j = 0
    for i in range(0, 512, 4):
        result[i] = ((data[j] & 0b11111100) >> 2) | ((data[j + 1] & 0b00000011) << 6)
        result[i + 1] = ((data[j + 2] & 0b11111100) >> 2) | ((data[j + 3] & 0b00000011) << 6)
        result[i + 2] = ((data[j + 4] & 0b11111100) >> 2) | ((data[j + 5] & 0b00000011) << 6)
        result[i + 3] = ((data[j + 6] & 0b11111100) >> 2) | ((data[j + 7] & 0b00000011) << 6)
        j = j + 8
    return result

def get_embed_params(embed_data):
    freq = 0
    shuffle = 0
    if embed_data[index_embed_phase] == value_embed_phase_sh:
        shuffle = 1
    elif embed_data[index_embed_phase] == value_embed_phase_ns:
        shuffle = 0
    else:
        print("Error embed format ")
        os.kill(0)
    if embed_data[index_embed_frequency_modulation] == value_embed_frequency_modulation_100:
        freq = 100
    elif embed_data[index_embed_frequency_modulation] == value_embed_frequency_modulation_80:
        freq = 80
    else:
        print("Error embed format!")
        os.kill(0)
    return freq, shuffle, embed_data[index_embed_frame_count]


def generate_coeff_map(w, h, fov_hor):
    result = np.zeros((h, w), dtype="float32")
    for y in range(0, h):
        for x in range(0, w):
            result[y][x] = get_z_cal_coeff(x, y, w, h, fov_hor)
    return result


def cal_depth16(phase_imgs, embed_data, coeff_map):
    freq, shuffle, framecount = get_embed_params(embed_data)
    print("freq %d, shuffle %d framecount %d"%(freq, shuffle, framecount))
    #print(shuffle)
    a0 = phase_imgs[0].astype("float32") #0
    a3 = phase_imgs[1].astype("float32") #270
    a1 = phase_imgs[2].astype("float32") #90
    a2 = phase_imgs[3].astype("float32") #180
    if shuffle == 1:
        a0, a1, a2, a3 = a2, a3, a0, a1
    phi = np.arctan2(a1 - a3, a0 - a2) #[-pi, pi]
    phi = np.map(lambda x: x + np.pi * 2 if x < 0 else x, phi)
    light_speed = 300 #Mega meter
    light_speed = light_speed/freq
    light_speed = light_speed*1000      # unit from m to mm. TODO: use 100 when using cm.
    depth = light_speed * phi/(np.pi*2.0)
    depth = depth*coeff_map
    depth = depth.astype("uint16")
    depth = np.clip(depth, 0, 0x1FFF)
    return [depth]


def pack_data_mipi10(data):
    shapex = np.shape(data)
    data = np.reshape((shapex[0] * shapex[1],))
    shape_r = (shapex[0] * 5 / 4, shapex[1])
    result = np.zeros(shape_r, dtype='unit8')
    j = 0
    for ix in range(0, len(data), 4):
        result[j] = (data[ix] >> 2) & 0xFF
        result[j + 1] = (data[ix + 1] >> 2) & 0xFF
        result[j + 2] = (data[ix + 2] >> 2) & 0xFF
        result[j + 3] = (data[ix + 3] >> 2) & 0xFF
        result[j + 4] = (data[ix] & 0x3) | ((data[ix + 1] & 0x3) << 2) \
                        | ((data[ix + 2] & 0x3) << 4) \
                        | ((data[ix + 3] & 0x3) << 6)
        j = j + 5
    return result


def pack_data_mipi12(data):
    shapex = np.shape(data)
    data = np.reshape((shapex[0] * shapex[1],))
    shape_r = (shapex[0] * 3 / 2, shapex[1])
    result = np.zeros(shape_r, dtype='uint8')
    j = 0
    for ix in range(0, len(data), 2):
        result[j] = data[ix] >> 4 & 0xFF;
        result[j + 1] = data[ix + 1] >> 4 & 0xFF
        result[j + 2] = (data[ix] & 0xF) | ((data[ix + 1] & 0xF) << 4)
    return result


def parse_depth16(args, buf, shape):
    result = np.frombuffer(buf, dtype='uint16')
    result = np.reshape(result, shape)
    result = np.bitwise_and(result, 0x1FFF)
    result = result.astype("float32")
    # max = 8191.0
    # min = 0.0
    # data = (data - min) / (max - min)
    return result


def parse_mipi10(args, buf, shape):
    res = unpack_data_10(buf, shape)
    res = np.frombuffer(res, dtype='uint16')
    res = res.astype("float32")
    res = np.reshape(res, shape)
    return res


def parse_mip12(args, buf, shape):
    res = unpack_data_12(buf, shape)
    res = res.astype("float32")
    res = np.reshape(res, shape)
    return res


def parse_raw16(args, buf, shape):
    res = np.frombuffer(buf, dtype="uint16")
    res = np.reshape(res[0:shape[0]*shape[1]], shape)
    res = res.astype("float32")
    return res

def parse_raw8(args, buf, shape):
    res = np.frombuffer(buf, dtype="uint8")
    res = np.reshape(res[0:shape[0]*shape[1]], shape)
    res = res.astype("float32")
    return res

def parse_raw32(args, buf, shape):
    result = np.frombuffer(buf, dtype="float32")
    result = np.reshape(result, shape)
    result = result.astype("float32")
    return result


def raw32_to_mipi10_bytes(args, raw32):
    result = raw32.astype("uint16")
    result = pack_data_mipi10(result)
    return [result]


def raw32_to_mipi12_bytes(args, raw32):
    result = raw32.astype("uint16")
    result = pack_data_mipi12(result)
    return [result]


def raw32_to_bayer16s_bytes(args, raw32):
    result = raw32.astype("uint16")
    result00 = result[0::2, 0::2]
    result01 = result[0::2, 1::2]
    result10 = result[1::2, 0::2]
    result11 = result[1::2, 1::2]
    return [result00, result01, result10, result11]


def raw32_to_raw16_bytes(args, raw32):
    result = raw32.astype("uint16")
    return [result]


def raw32_to_png_bytes(args, raw32, bits):
    if bits == 0:
        max = np.max(raw32)
    else:
        max = float(1<<bits -1)
    raw32 = raw32/max
    raw32 = raw32*255
    raw32 = raw32.astype("uint8")
    #print(np.shape(raw32))imencode
    res, image = cv2.imencode(".png", raw32)
    return [image]


def main_process(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-if", "--input_format",
                        help="the format of the input data.(depth16, mipi10, mipi12, raw8, raw10"
                             ", raw12, raw14, raw16, raw32f)", required=True)
    parser.add_argument("-of", "--output_format",
                        help="the format of the output data.(mipi10, mipi12, bayer16, raw16, raw32, png, freq)",
                        required=True)
    parser.add_argument("-c", "--col", help="the width of the image", required=True)
    parser.add_argument("-r", "--row", help="the height of the image", required=True)
    parser.add_argument("-i", "--input", help="the input file pattern(match by *)", required=True)
    parser.add_argument("-o", "--output", help="the output directory", required=True)
    parser.add_argument("-m", "--metadata", help="the meta data 0: inline,  1: calibration dump, _VC(0)_DT(Embedded_Data).raw  2: test dump _VC(0).raw", required=False)
    print(sys.argv)
    namespace = parser.parse_args()
    shape = (int(namespace.row), int(namespace.col))
    #embed line is 1 for depth.
    if not os.path.exists(namespace.output):
        os.makedirs(namespace.output)
    print(namespace.row)
    print(namespace.col)
    print(namespace.output)
    print(namespace.input)
    #cv2.imshow("win", coeff_map)
    #cv2.waitKey()
    for file in glob.glob(namespace.input):
        print("namespace.input.file = " + file)
        f = open(file, mode='r+b')
        subpaths = file.split("/")
        dir = os.path.dirname(file)
        f_name = subpaths[len(subpaths) - 1]
        buf = f.read()
        res = []
        bit_count = 0
        emb_data = None
        #print(type(buf))

        test_cal = "_VC(0).raw"
        suffix_cal = "_VC(0)_DT(Embedded_Data).raw"
        if f_name.endswith(suffix_cal) or f_name.endswith(test_cal):
            continue
        if namespace.metadata == None:
            namespace.metadata = 0
        f_name_emb = None
        if int(namespace.metadata) == 1:
            f_name_emb = f_name[0: len(f_name) - 4] + suffix_cal
            f_name_emb = dir + "/" + f_name_emb
        elif int(namespace.metadata) == 2:
            f_name_emb = f_name[0: len(f_name) - 4] + test_cal
            f_name_emb = dir + "/" + f_name_emb
        if namespace.input_format == "depth16":
            print(len(buf))
            print(file)
            res = parse_depth16(namespace, buf, shape)
            bit_count = 13
        elif namespace.input_format == "mipi10":
            res = parse_mipi10(namespace, buf, shape)
            emb_data = get_embeded_data(namespace.input_format, namespace.metadata, buf, f_name_emb)
            bit_count = 10
        elif namespace.input_format == "mipi12":
            res = parse_mip12(namespace, buf, shape)
            bit_count = 12
        elif namespace.input_format == "raw8":
            #emb_data = get_embeded_data_raw10(buf)
            res = parse_raw8(namespace, buf, shape)
            bit_count = 8
        elif namespace.input_format == "raw10":
            emb_data = get_embeded_data(namespace.input_format, namespace.metadata, buf, f_name_emb)
            res = parse_raw16(namespace, buf, shape)
            bit_count = 10
        elif namespace.input_format == "raw12":
            res = parse_raw16(namespace, buf, shape)
            bit_count = 12
        elif namespace.input_format == "raw14":
            res = parse_raw16(namespace, buf, shape)
            bit_count = 14
        elif namespace.input_format == "raw16":
            res = parse_raw16(namespace, buf, shape)
            bit_count = 16
        elif namespace.input_format == "bayer16":
            res = parse_raw16(namespace, buf, shape)
            bit_count = 16
        elif namespace.input_format == "raw32":
            res = parse_raw32(namespace, buf, shape)
            bit_count = 0

        #print(np.shape(res))
        if namespace.output_format == "depth16":
            #remove embed line for 33D tof.

            depthW = 0
            depthH = 0
            if int(namespace.metadata) == 0:
                depthW = int(shape[1] / 2)
                depthH = int(shape[0] / 2 - 1)
                res = raw32_to_bayer16s_bytes(namespace, res[2::])
            else:
                depthW = int(shape[1] / 2)
                depthH = int(shape[0] / 2)
                res = raw32_to_bayer16s_bytes(namespace, res)
            print(" depthW %d depthH %d "%(depthW, depthH))
            z_calibration_coeff_map = generate_coeff_map(depthW, depthH, FOV_HOR)

            print("namespace.input = " + namespace.input)
            print("coeff_map = " + str(np.shape(z_calibration_coeff_map)))
            res = cal_depth16(res, emb_data, z_calibration_coeff_map)
            freq, shuffle, frame_count = get_embed_params(emb_data)

            if True:
                cal_dump_png = raw32_to_png_bytes(namespace, z_calibration_coeff_map, 1)
                print(type(cal_dump_png))
                f = open(namespace.output + "/" + "z_calibration_coeff_map.png", "wb")
                f.write(cal_dump_png[0])
                f.close()

        elif namespace.output_format == "mipi10":
            res = raw32_to_mipi10_bytes(namespace, res)
        elif namespace.output_format == "mipi12":
            res = raw32_to_mipi12_bytes(namespace, res)
        elif namespace.output_format == "bayer16":
            res = raw32_to_bayer16s_bytes(namespace, res)
        elif namespace.output_format == "raw16":
            res = raw32_to_raw16_bytes(namespace, res)
        elif namespace.output_format == "raw32":
            pass
        elif namespace.output_format == "png":
            res = raw32_to_png_bytes(namespace, res, bit_count)
        else:
            print("eorror format %s"%namespace.output_format)
            exit(-1)

        if not os.path.exists(namespace.output):
            os.makedirs(namespace.output)

        bayer_index = 0
        for res_i in res:
            fw = open(namespace.output + "/" + f_name + "_" + str(bayer_index) + "." + namespace.output_format, mode='wb')
            #print(np.shape(res_i))
            fw.write(res_i.tobytes())
            fw.close()
            bayer_index = bayer_index + 1
        if namespace.output_format == "depth16":
            fw = open(namespace.output + "/" + f_name + ".meta", mode='wt')
            fw.write("freq = " + str(freq) + "\n\r")
            fw.write("shuffle = " + str(shuffle) + "\n\r")
            fw.write("frame_count = " + str(shuffle) + "\n\r")

if __name__ == "__main__":

    main_process(sys.argv)
