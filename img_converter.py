import glob
import numpy as np
import argparse
import sys
import os
import re


def check_file_size(file, shape, input_format, suffix):
    real_size = os.path.getsize(file)
    if input_format == "mipi10" or suffix == ".RAWMIPI10" or suffix == ".RAWMIPI":
        ideal_size = shape[0] * shape[1] * 5 / 4
    elif input_format == "mipi12" or suffix == ".RAWMIPI12":
        ideal_size = shape[0] * shape[1] * 3 / 2
    else:
        print("error format %s" % input_format)
        exit(-1)
    if real_size == ideal_size:
        print("[real_size = ideal_size]\n real_size = %d B,ideal_size = %d B" % (real_size, ideal_size))
        return 1
    elif real_size > ideal_size:
        print("[real_size > ideal_size]\n real_size = %d B,ideal_size = %d B\n please check!" % (real_size, ideal_size))
        return -1
    elif real_size < ideal_size:
        print("[real_size < ideal_size]\n real_size = %d B,ideal_size = %d B\n please check!" % (real_size, ideal_size))
        return -1


### raw_12 --> unpacked : 2 pixels occupy 3 bytes
#### |a04 a05 a06 a07 a08 a09 a10 a11|b04 b05 b06 b07 b08 b09 b10 b11|a00 a01 a02 a03 b00 b01 b02 b03|
#### |-----------data[0]-------------|------------data[1]------------|----------data[2]--------------|
#### |a00 a01 a02 a03 a04 a05 a06 a07 a08 a09 a10 a11|b00 b01 b02 b03 b04 b05 b06 b07 b08 b09 b10 b11|
#### |-------------------result[0]-------------------|-----------------result[1]---------------------|
def unpack_data_12(data, shape):
    result = np.zeros(shape[0] * shape[1], "uint16")
    j = 0
    for ix in range(0, len(data), 3):
        result[j] = (data[ix] << 4) | ((data[ix + 2]) >> 0 & 0xF)
        result[j + 1] = (data[ix + 1] << 4) | ((data[ix + 2] >> 4) & 0xF)
        j = j + 2
    return result


### raw_10--> unpacked ： 4 pixels occupy 5 bytes
#### |a02 a03 a04 a05 a06 a07 a08 a09|b02 b03 b04 b05 b06 b07 b08 b09|c02 c03 c04 c05 c06 c07 c08 c09|d02 d03 d04 d05 d06 d07 d08 d09|a00 a01 b00 b01 c00 c01 d00 d01|
#### |-----------data[0]-------------|------------data[1]------------|----------data[2]--------------|----------data[3]--------------|----------data[4]--------------|
#### |a00 a01 a02 a03 a04 a05 a06 a07 a08 a09|b00 b01 b02 b03 b04 b05 b06 b07 b08 b09|c00 c01 c02 c03 c04 c05 c06 c07 c08 c09|d00 d01 d02 d03 d04 d05 d06 d07 d08 d09|
#### |----------------result[0]--------------|--------------result[1]----------------|--------------result[2]----------------|--------------result[3]----------------|
def unpack_data_10(data, shape):
    result = np.zeros(shape[0] * shape[1], "uint16")
    j = 0
    for ix in range(0, len(data), 5):
        result[j] = (data[ix] << 2) | ((data[ix + 4]) >> 0 & 0x3)
        result[j + 1] = (data[ix + 1] << 2) | ((data[ix + 4] >> 2) & 0x3)
        result[j + 2] = (data[ix + 2] << 2) | ((data[ix + 4] >> 4) & 0x3)
        result[j + 3] = (data[ix + 3] << 2) | ((data[ix + 4] >> 6) & 0x3)
        j = j + 4
    return result


### unpacked --> raw_10
def pack_data_mipi10(data):
    shapex = np.shape(data)
    data = np.reshape((shapex[0] * shapex[1],))
    shape_r = (shapex[0] * 5 / 4, shapex[1])  # width*5/4 (for 8-bit align)
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


### unpacked --> raw_12
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


def parse_mipi10(args, buf, shape):
    res = unpack_data_10(buf, shape)
    res = np.frombuffer(res, dtype='uint16')  # read buffer and convert to nparray
    res = res.astype("float32")
    res = np.reshape(res, shape)
    return res


def parse_mip12(args, buf, shape):
    res = unpack_data_12(buf, shape)
    res = res.astype("float32")
    res = np.reshape(res, shape)
    return res


def raw32_to_mipi10_bytes(args, raw32):
    result = raw32.astype("uint16")
    result = pack_data_mipi10(result)
    return [result]


def raw32_to_mipi12_bytes(args, raw32):
    result = raw32.astype("uint16")
    result = pack_data_mipi12(result)
    return [result]


def raw32_to_raw16_bytes(args, raw32):
    result = raw32.astype("uint16")
    return [result]


def main_process(argv):
    parser = argparse.ArgumentParser(description="Image Format Convertor")
    parser.add_argument("-if", "--input_format",
                        help="the format of the input data.(mipi10, mipi12)\n[default] from filename", required=False)
    parser.add_argument("-of", "--output_format",
                        help="the format of the output data.(mipi10, mipi12, raw16)\n[default] unpacked raw16",
                        required=False)
    parser.add_argument("-c", "--col", help="the width of the image", required=False)
    parser.add_argument("-r", "--row", help="the height of the image", required=False)
    parser.add_argument("-i", "--input", help="the input file pattern(match by *)", required=True)
    parser.add_argument("-o", "--output", help="the output directory")
    print(sys.argv)
    namespace = parser.parse_args()

    if not os.path.exists(namespace.output):
        os.makedirs(namespace.output)

    for file in glob.glob(namespace.input):
        print('namespace.input.file = ' + file)
        f = open(file, mode='r+b')
        sub_paths = file.split("/")
        dir = os.path.dirname(file)
        f_name = sub_paths[len(sub_paths) - 1]
        print(f_name)

        suffix = os.path.splitext(f_name)[-1]
        print(suffix)

        weight_key = re.compile(r'\_w\[(.+?)\]\_')
        height_key = re.compile(r'\_h\[(.+?)\]\_')
        weight = re.findall(weight_key, f_name)
        weight = int(weight[0])
        height = re.findall(height_key, f_name)
        height = int(height[0])
        print("w=%d\th=%d from filename" % (weight, height))

        if namespace.row is None:
            namespace.row = weight
        elif int(namespace.row) == weight:
            print("input correct weight=[%d]" % int(namespace.row))
        else:
            print("input weight maybe wrong, plz check!")
            return 0

        if namespace.col is None:
            namespace.col = height
        elif int(namespace.col) == height:
            print("input correct height = [%d]" % int(namespace.col))
        else:
            print("input height maybe wrong, plz check!")
            return 0

        if namespace.input_format is None:
            print("input format =[%s] from filename" % suffix)
        else:
            print("output format=[%s]" % namespace.input_format)
        if namespace.output_format is None:
            namespace.output_format = "raw16"
            print("output format default=[%s]" % namespace.output_format)
        else:
            print("output format=[%s]" % namespace.output_format)

        shape = (int(namespace.row), int(namespace.col))
        print('input shape[%d %d]' % (shape[0], shape[1]))

        check_sz = check_file_size(file, shape, namespace.input_format, suffix)
        if check_sz != 1:
            print("%s size is not matched!!!" % f_name)
            return 0

        buf = f.read()
        res = []
        bit_count = 0
        print(type(buf))
        if namespace.input_format == "mipi10" or suffix == ".RAWMIPI10" or suffix == ".RAWMIPI":
            res = parse_mipi10(namespace, buf, shape)
            bit_count = 10
        elif namespace.input_format == "mipi12" or suffix == ".RAWMIPI12":
            res = parse_mip12(namespace, buf, shape)
            bit_count = 12
        else:
            print("input format %s is not supported" % namespace.input_format)
            exit(-1)

        print(np.shape(res))
        if namespace.output_format == "mipi10":
            res = raw32_to_mipi10_bytes(namespace, res)
        elif namespace.output_format == "mipi12":
            res = raw32_to_mipi12_bytes(namespace, res)
        elif namespace.output_format == "raw16":  # 16 bits occupy 2 bytes (1 uint16) exactly
            res = raw32_to_raw16_bytes(namespace, res)
        else:
            print("error format %s" % namespace.output_format)
            exit(-1)

        if not os.path.exists(namespace.output):
            os.makedirs(namespace.output)

        for res_i in res:
            fw = open(namespace.output + "/" + f_name + "." + namespace.output_format + ".raw", mode='wb')
            fw.write(res_i.tobytes())
            fw.close()
    print("Finish Converter!")


if __name__ == "__main__":
    main_process(sys.argv)
