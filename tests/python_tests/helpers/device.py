from ttlens.tt_lens_init import init_ttlens
from ttlens.tt_lens_lib import write_to_device, read_words_from_device, run_elf
from helpers import *

ELF_LOCATION = "../build/elf/"

def run_elf_files(testname, run_brisc=True):

    if run_brisc == True:
        run_elf(f"{ELF_LOCATION}brisc.elf", "0,0", risc_id=0)

    for i in range(3):
        run_elf(f"{ELF_LOCATION}{testname}_trisc{i}.elf", "0,0", risc_id=i + 1)

def write_stimuli_to_l1(buffer_A, buffer_B, stimuli_format, tile_cnt = 1):

    if tile_cnt != 1:   
        buffer_B_address = 0x1a000 + 1024*tile_cnt
        buffer_A_address = 0x1a000
    else:
        buffer_B_address = 0x1b000
        buffer_A_address = 0x1a000

    if stimuli_format == "Float16_b":
        write_to_device("0,0", buffer_A_address, pack_bfp16(buffer_A))
        write_to_device("0,0", buffer_B_address, pack_bfp16(buffer_B))    
    elif stimuli_format == "Float16":
        write_to_device("0,0", buffer_A_address, pack_fp16(buffer_A))
        write_to_device("0,0", buffer_B_address, pack_fp16(buffer_B))
    elif stimuli_format == "Bfp8_b":
        write_to_device("0,0", buffer_A_address, pack_bfp8_b(buffer_A))
        write_to_device("0,0", buffer_B_address, pack_bfp8_b(buffer_B))
    elif stimuli_format == "Int32":
        write_to_device("0,0", buffer_A_address, pack_int32(buffer_A))
        write_to_device("0,0", buffer_B_address, pack_int32(buffer_B))
    elif stimuli_format == "Float32":
        write_to_device("0,0", buffer_A_address, pack_fp32(buffer_A))
        write_to_device("0,0", buffer_B_address, pack_fp32(buffer_B))

def get_result_from_device(format,read_data_bytes):
    if(format == "Float16"):
        res_from_L1 = unpack_fp16(read_data_bytes)
    elif(format == "Float16_b"):
        res_from_L1 = unpack_bfp16(read_data_bytes)
    elif( format == "Bfp8_b"):
        res_from_L1 = unpack_bfp8_b(read_data_bytes)
    elif( format == "Float32"):
        res_from_L1 = unpack_float32(read_data_bytes)
    elif( format == "Int32"):
        res_from_L1 = unpack_int32(read_data_bytes)

    return res_from_L1