import pytest
import torch
import os
from ttlens.tt_lens_init import init_ttlens
from ttlens.tt_lens_lib import write_to_device, read_words_from_device, run_elf
import math
from helpers import *

ELF_LOCATION = "../build/elf/"

def run_elf_files(testname, run_brisc=True):

    if run_brisc == True:
        run_elf(f"{ELF_LOCATION}brisc.elf", "0,0", risc_id=0)

    for i in range(3):
        run_elf(f"{ELF_LOCATION}{testname}_trisc{i}.elf", "0,0", risc_id=i + 1)

def generate_stimuli(stimuli_format):

    # for simplicity stimuli is only 256 numbers
    # since sfpu operates only on part of dest
    #srcA = torch.full((256,), 2, dtype=format_dict[stimuli_format])
    srcA = torch.rand(256, dtype=format_dict[stimuli_format]) + 0.5
    return srcA

def generate_golden(operation, operand1, data_format):
    tensor1_float = operand1.clone().detach().to(format_dict[data_format])

    res = []

    if(operation == "sqrt"):
        for number in tensor1_float.tolist():
            res.append(math.sqrt(number))
    elif(operation == "square"): 
        for number in tensor1_float.tolist():
            res.append(number*number)
    elif(operation == "log"):
        for number in tensor1_float.tolist():
            if(number != 0):
                res.append(math.log(number))
            else:
                res.append(float('nan'))
    else:
        raise ValueError("Unsupported operation!")

    return res

def write_stimuli_to_l1(buffer_A, stimuli_format):
    if stimuli_format == "Float16_b":
        write_to_device("0,0", 0x1b000, pack_bfp16(buffer_A))
    elif stimuli_format == "Float16":
        write_to_device("0,0", 0x1b000, pack_fp16(buffer_A))

@pytest.mark.parametrize("format", ["Float16"])
@pytest.mark.parametrize("testname", ["eltwise_unary_sfpu_test"])
@pytest.mark.parametrize("mathop", ["square", "sqrt", "log"])
def test_all(format, mathop, testname):
    #context = init_debuda()
    src_A = generate_stimuli(format)
    golden = generate_golden(mathop, src_A, format)
    write_stimuli_to_l1(src_A, format)

    make_cmd = f"make --silent format={format_args_dict[format]} mathop={mathop_args_dict[mathop]} testname={testname}"
    os.system(f"cd .. && {make_cmd}")

    run_elf_files(testname)

    if(format == "Float16" or format == "Float16_b"):
        read_words_cnt = len(src_A)//2
    
    read_data = read_words_from_device("0,0", 0x1a000, word_count=read_words_cnt)
    
    read_data_bytes = flatten_list([int_to_bytes_list(data) for data in read_data])
    
    if (format == "Float16_b"):
        res_from_L1 = unpack_bfp16(read_data_bytes)
    elif (format == "Float16"):
        res_from_L1 = unpack_fp16(read_data_bytes)
    
    assert len(res_from_L1) == len(golden)

    os.system("cd .. && make clean")

    # Mailbox checks
    assert read_words_from_device("0,0", 0x19FF4, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'
    assert read_words_from_device("0,0", 0x19FF8, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'
    assert read_words_from_device("0,0", 0x19FFC, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'

    golden_tensor = torch.tensor(golden, dtype=format_dict[format] if format in ["Float16", "Float16_b"] else torch.bfloat16)
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[format] if format in ["Float16", "Float16_b"] else torch.bfloat16)

    if(format == "Float16_b" or format == "Float16"):
        atol = 0.05
        rtol = 0.1

    for i in range(len(golden)):
        assert torch.isclose(golden_tensor[i],res_tensor[i], rtol = rtol, atol = atol), f"Failed at index {i} with values {golden[i]} and {res_from_L1[i]}"