import pytest
import torch
import os
import itertools
from helpers import *

def generate_math_kernels(length):
    return list(itertools.product([1, 2, 3], repeat=length))

def generate_golden(operations, operand1, operand2, data_format):
    tensor1_float = operand1.clone().detach().to(format_dict[data_format])
    tensor2_float = operand2.clone().detach().to(format_dict[data_format])
    
    res = []

    for op in operations:
        if(op==1):
            res_tmp = tensor1_float + tensor2_float
        elif(op==2):
            res_tmp = tensor1_float - tensor2_float
        elif(op==3):
            res_tmp = tensor1_float * tensor2_float
        else:
            raise ValueError("Unsupported operation!")
        
        res.append(res_tmp.tolist())
    
    return res

pack_addresses = [0x1a000,0x1d000, 0x1e000] #, 0x1f000, 0x20000, 0x21000, 0x22000, 0x23000, 0x24000, 0x25000]


@pytest.mark.parametrize("length", range(1,len(pack_addresses)+1))
@pytest.mark.parametrize("format", ["Float16_b"])
@pytest.mark.parametrize("testname", ["multiple_ops_test"])
def test_multiple_kernels(format, testname,length):

    unpack_kernels = [2]*length
    pack_kernels = [1]*length

    math_kernels_list = generate_math_kernels(length)
    math_kernels_list = [list(kernel) for kernel in math_kernels_list]

    for math_kernels in math_kernels_list:

        unpack_kerns_formatted = format_kernel_list(unpack_kernels)
        math_kerns_formatted = format_kernel_list(math_kernels)
        pack_kerns_formatted = format_kernel_list(pack_kernels)
        pack_addresses_formatted = format_kernel_list(pack_addresses, as_hex=True)

        #context = init_debuda()
        src_A, src_B = generate_stimuli(format)
        golden = generate_golden(math_kernels, src_A, src_B, format)
        write_stimuli_to_l1(src_A, src_B, format)

        make_cmd = f"make --silent format={format_args_dict[format]} testname={testname}"
        make_cmd += " unpack_kern_cnt="+ str(len(unpack_kernels))+ " unpack_kerns="+unpack_kerns_formatted
        make_cmd += " math_kern_cnt="+ str(len(math_kernels))+ " math_kerns="+math_kerns_formatted
        make_cmd += " pack_kern_cnt="+ str(len(pack_kernels))+ " pack_kerns="+pack_kerns_formatted
        make_cmd += " pack_addr_cnt="+ str(len(pack_addresses))+ " pack_addrs="+pack_addresses_formatted
        os.system(f"cd .. && {make_cmd}")

        run_elf_files(testname)

        os.system("cd .. && make clean")

        # Mailbox checks
        assert read_words_from_device("0,0", 0x19FF4, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'
        assert read_words_from_device("0,0", 0x19FF8, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'
        assert read_words_from_device("0,0", 0x19FFC, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'

        for index in range(len(golden)):
            read_words_cnt = len(src_A) // (2 if format in ["Float16", "Float16_b"] else 1)
            read_data = read_words_from_device("0,0", pack_addresses[index], word_count=read_words_cnt)
            read_data_bytes = flatten_list([int_to_bytes_list(data) for data in read_data])
            res_from_L1 = unpack_bfp16(read_data_bytes) if format == "Float16_b" else unpack_fp16(read_data_bytes)
            curr_golden = golden[index]

            assert len(res_from_L1) == len(curr_golden)
            print("Checking all elements of golden at index: ", index)

            golden_tensor = torch.tensor(curr_golden, dtype=format_dict[format] if format in ["Float16", "Float16_b"] else torch.bfloat16)
            res_tensor = torch.tensor(res_from_L1, dtype=format_dict[format] if format in ["Float16", "Float16_b"] else torch.bfloat16)

            if(format == "Float16_b" or format == "Float16"):
                atol = 0.05
                rtol = 0.1

        for i in range(len(curr_golden)):
            assert torch.isclose(golden_tensor[i],res_tensor[i], rtol = rtol, atol = atol), f"Failed at index {i} with values {curr_golden[i]} and {res_from_L1[i]}"
        
        _ , pcc = comp_pcc(golden_tensor, res_tensor, pcc=0.99) 
        assert pcc > 0.99