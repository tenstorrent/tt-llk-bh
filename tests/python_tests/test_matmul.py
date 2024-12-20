import pytest
import torch
import os
import numpy as np
from ttlens.tt_lens_init import init_ttlens
from ttlens.tt_lens_lib import write_to_device, read_words_from_device, run_elf
from helpers import *

ELF_LOCATION = "../build/elf/"
torch.set_printoptions(precision=2, linewidth=800, threshold=100000, sci_mode=False)

def run_elf_files(testname, run_brisc=True):

    if run_brisc == True:
        run_elf(f"{ELF_LOCATION}brisc.elf", "0,0", risc_id=0)

    for i in range(3):
        run_elf(f"{ELF_LOCATION}{testname}_trisc{i}.elf", "0,0", risc_id=i + 1)

def generate_golden(operand1, operand2, data_format):
    A_float = operand1.clone().detach().to(format_dict[data_format])
    B_float = operand2.clone().detach().to(format_dict[data_format])

    A_untilized = untilize(A_float,data_format)
    B_untilized = untilize(B_float,data_format)

    result = torch.matmul(A_untilized, B_untilized )

    result = tilize(result)
    return result

def write_stimuli_to_l1(buffer_A, buffer_B, stimuli_format):
    if stimuli_format == "Float16_b":
        write_to_device("0,0", 0x1b000, pack_bfp16(buffer_A))
        write_to_device("0,0", 0x1c000, pack_bfp16(buffer_B))    
    elif stimuli_format == "Float16":
        write_to_device("0,0", 0x1b000, pack_fp16(buffer_A))
        write_to_device("0,0", 0x1c000, pack_fp16(buffer_B))

@pytest.mark.parametrize("format", ["Float16_b"])#, "Float16"])
@pytest.mark.parametrize("testname", ["matmul_test"])
def test_all(format, testname):

    #context = init_debuda()
    src_A, src_B = generate_stimuli(format)
    golden_tensor = generate_golden(src_A, src_B, format)
    # print("$"*80)
    # print(src_A[0] , src_A[255], src_A[256], src_A[511], src_A[512])


    print(src_A.view(32,32))
    print(src_B.view(32,32))

    write_stimuli_to_l1(src_A, src_B, format)

    make_cmd = f"make format={format_args_dict[format]} testname={testname}"
    os.system(f"cd .. && {make_cmd}")

    run_elf_files(testname)

    read_words_cnt = len(src_A) // (2 if format in ["Float16", "Float16_b"] else 1)
    read_data = read_words_from_device("0,0", 0x1a000, word_count=read_words_cnt)
    
    read_data_bytes = flatten_list([int_to_bytes_list(data) for data in read_data])
    
    res_from_L1 = unpack_bfp16(read_data_bytes) if format == "Float16_b" else unpack_fp16(read_data_bytes)

    #assert len(res_from_L1) == len(golden)

    os.system("cd .. && make clean")

    # Mailbox checks
    assert read_words_from_device("0,0", 0x19FF4, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'
    assert read_words_from_device("0,0", 0x19FF8, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'
    assert read_words_from_device("0,0", 0x19FFC, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'

    #golden_tensor = torch.tensor(golden, dtype=format_dict[format] if format in ["Float16", "Float16_b"] else torch.bfloat16)
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[format] if format in ["Float16", "Float16_b"] else torch.bfloat16)
    #res_tensor = untilize(res_tensor,format)
    print(untilize(res_tensor,format).view(32,32))
    print("\n" *10 )
    print(untilize(golden_tensor,format).view(32,32))

    if(format == "Float16_b" or format == "Float16"):
        atol = 0.2
        rtol = 0.1
    elif(format == "Bfp8_b"):
        atol = 0.4
        rtol = 0.3

    rel_errs = []

    #for i in range(len(golden)):
        #rel_errs.append(relative_error(golden_tensor[i], res_tensor[i]))
        #assert torch.isclose(golden_tensor[i],res_tensor[i], rtol = rtol, atol = atol), f"Failed at index {i} with values {golden[i]} and {res_from_L1[i]}"

    # print("#"*50)
    # print("PCC")
    # print(pearson_correlation(golden_tensor,res_tensor).item())

    print(comp_pcc(golden_tensor, res_tensor, pcc=0.99))

    assert 1==0