from ttlens.tt_lens_init import init_ttlens
from ttlens.tt_lens_lib import write_to_device, read_words_from_device, run_elf
from helpers import *

ELF_LOCATION = "../build/elf/"

def run_elf_files(testname, run_brisc=True):

    if run_brisc == True:
        run_elf(f"{ELF_LOCATION}brisc.elf", "0,0", risc_id=0)

    for i in range(3):
        run_elf(f"{ELF_LOCATION}{testname}_trisc{i}.elf", "0,0", risc_id=i + 1)

def write_stimuli_to_l1(buffer_A, buffer_B, stimuli_format):
    if stimuli_format == "Float16_b":
        write_to_device("0,0", 0x1b000, pack_bfp16(buffer_A))
        write_to_device("0,0", 0x1c000, pack_bfp16(buffer_B))    
    elif stimuli_format == "Float16":
        write_to_device("0,0", 0x1b000, pack_fp16(buffer_A))
        write_to_device("0,0", 0x1c000, pack_fp16(buffer_B))
    elif stimuli_format == "Bfp8_b":
        write_to_device("0,0", 0x1b000, pack_bfp8_b(buffer_A))
        write_to_device("0,0", 0x1c000, pack_bfp8_b(buffer_B))
    elif stimuli_format == "Int32":
        write_to_device("0,0", 0x1b000, pack_int32(buffer_A))
        write_to_device("0,0", 0x1c000, pack_int32(buffer_B))
    elif stimuli_format == "Float32":
        write_to_device("0,0", 0x1b000, pack_fp32(buffer_A))
        write_to_device("0,0", 0x1c000, pack_fp32(buffer_B))
