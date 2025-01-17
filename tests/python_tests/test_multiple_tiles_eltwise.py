import pytest
import torch
import os
from helpers import *

def generate_golden(op, operand1, operand2, data_format):
    tensor1_float = operand1.clone().detach().to(format_dict[data_format])
    tensor2_float = operand2.clone().detach().to(format_dict[data_format])

    if(op==1):
        res = tensor1_float + tensor2_float
    elif(op==2):
        res = tensor1_float - tensor2_float
    elif(op==3):
        res = tensor1_float * tensor2_float
    else:
        raise ValueError("Unsupported operation!")
    
    return res.tolist()

@pytest.mark.parametrize("mathop", range(1,4))
@pytest.mark.parametrize("tile_cnt", range(1,4))
@pytest.mark.parametrize("format", ["Float16_b", "Float16"])
@pytest.mark.parametrize("dest_acc", ["","DEST_ACC"])
@pytest.mark.parametrize("testname", ["multiple_tiles_eltwise_test"])
def test_multiple_kernels(format, testname, tile_cnt, mathop, dest_acc):

    # prepare setup for running kernels

    pack_start_address = 0x1a000 + 2*4096*tile_cnt
    pack_addresses = [pack_start_address + 0x1000 * i for i in range(tile_cnt)]

    unpack_kernels = [2] * tile_cnt
    pack_kernels = [1] * tile_cnt
    math_kernels = [mathop] * tile_cnt

    unpack_kerns_formatted = format_kernel_list(unpack_kernels)
    math_kerns_formatted = format_kernel_list(math_kernels)
    pack_kerns_formatted = format_kernel_list(pack_kernels)
    pack_addresses_formatted = format_kernel_list(pack_addresses, as_hex=True)

    #context = init_debuda()
    src_A, src_B = generate_stimuli(format,tile_cnt = tile_cnt)
    golden = generate_golden(mathop,src_A,src_B,format)
    write_stimuli_to_l1(src_A,src_B,format,tile_cnt)

    make_cmd = f"make --silent format={format_args_dict[format]} testname={testname} dest_acc={dest_acc}"
    make_cmd += " unpack_kern_cnt="+ str(len(unpack_kernels))+ " unpack_kerns="+unpack_kerns_formatted
    make_cmd += " math_kern_cnt="+ str(len(math_kernels))+ " math_kerns="+math_kerns_formatted
    make_cmd += " pack_kern_cnt="+ str(len(pack_kernels))+ " pack_kerns="+pack_kerns_formatted
    make_cmd += " pack_addr_cnt="+ str(len(pack_addresses))+ " pack_addrs="+pack_addresses_formatted
    make_cmd += " unpack_a_addr_cnt="+str(tile_cnt)

    os.system(f"cd .. && {make_cmd}")

    run_elf_files(testname)

    os.system("cd .. && make clean")

    assert read_words_from_device("0,0", 0x19FF4, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'
    assert read_words_from_device("0,0", 0x19FF8, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'
    assert read_words_from_device("0,0", 0x19FFC, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'

    #check resluts from multiple tiles

    read_words_cnt = len(src_A) // (2 if format in ["Float16", "Float16_b"] else 1)
    read_data = read_words_from_device("0,0", pack_start_address, word_count=read_words_cnt*tile_cnt)
    read_data_bytes = flatten_list([int_to_bytes_list(data) for data in read_data])
    res_from_L1 = unpack_bfp16(read_data_bytes) if format == "Float16_b" else unpack_fp16(read_data_bytes)

    if(format == "Float16" or format == "Float16_b"):
        chunk_size = 512
    else:
        chunk_size = 1024
    
    res_sublists = [res_from_L1[i:i + chunk_size] for i in range(0, len(res_from_L1), chunk_size)]

    golden_tensor = torch.tensor(golden, dtype=format_dict[format] if format in ["Float16", "Float16_b"] else torch.bfloat16)
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[format] if format in ["Float16", "Float16_b"] else torch.bfloat16)

    if(format == "Float16_b" or format == "Float16"):
        atol = 0.05
        rtol = 0.1
    elif(format == "Bfp8_b"):
        atol = 0.4
        rtol = 0.3

    for sublist in res_sublists:
        for i in range(len(sublist)):  
            assert torch.isclose(torch.tensor(res_from_L1[i]),torch.tensor(golden[i]), rtol = rtol, atol = atol), f"Failed at index {i} with values {golden[i]} and {res_from_L1[i]}"
