import pytest
import torch
import os
from helpers import *

def generate_golden(operation, operand1, operand2, data_format):
    if( data_format == "Float16" or data_format == "Float16_b"):
        tensor1_float = operand1.clone().detach().to(format_dict[data_format])
        tensor2_float = operand2.clone().detach().to(format_dict[data_format])
    else:
        tensor1_float = operand1.clone().detach().to(format_dict["Float16_b"])
        tensor2_float = operand2.clone().detach().to(format_dict["Float16_b"])
    
    operations = {
        "elwadd": tensor1_float + tensor2_float,
        "elwsub": tensor1_float - tensor2_float,
        "elwmul": tensor1_float * tensor2_float,
        "elwdiv": tensor1_float / tensor2_float
    }
    
    if operation not in operations:
        raise ValueError("Unsupported operation!")

    return operations[operation].tolist()

formats = ["Bfp8_b", "Float16_b", "Float16"]
formats = ["Bfp8_b"]
mathops = ["elwadd", "elwsub", "elwmul"]
mathops = ["elwadd"]
@pytest.mark.parametrize("input_format", formats)
@pytest.mark.parametrize("output_format", formats)
@pytest.mark.parametrize("testname", ["eltwise_binary_test"])
@pytest.mark.parametrize("mathop", mathops)
@pytest.mark.parametrize("dest_acc", [""])#, "DEST_ACC"])
def test_all(input_format, output_format,mathop, testname, dest_acc):
    #context = init_debuda()
    if input_format != output_format:
        pytest.skip("")
    src_A, src_B = generate_stimuli(input_format)
    print("SRC_A = ", src_A)
    print("SRC_B = ", src_B)
    golden = generate_golden(mathop, src_A, src_B, output_format)
    write_stimuli_to_l1(src_A, src_B, input_format)

    test_config = {
        "input_format": input_format,
        "output_format": output_format,
        "testname": testname,
        "dest_acc": dest_acc,
        "mathop": mathop
    }

    make_cmd = generate_make_command(test_config)
    os.system(f"cd .. && {make_cmd}")

    run_elf_files(testname)
    
    res_from_L1 = collect_results(output_format,src_A)
    print()
    print("GOLDEN = ", golden)
    
    assert len(res_from_L1) == len(golden)

    os.system("cd .. && make clean")

    # Mailbox checks
    assert read_words_from_device("0,0", 0x19FF4, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'
    assert read_words_from_device("0,0", 0x19FF8, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'
    assert read_words_from_device("0,0", 0x19FFC, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'

    if(output_format == "Float16_b" or output_format == "Float16"):
        atol = 0.05
        rtol = 0.1
    elif(output_format == "Bfp8_b"):
        atol = 0.1
        rtol = 0.2

    golden_tensor = torch.tensor(golden, dtype=format_dict[output_format] if output_format in ["Float16", "Float16_b"] else torch.bfloat16)
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[output_format] if output_format in ["Float16", "Float16_b"] else torch.bfloat16)
    print("RESULT TENSOR = ", res_tensor)
    print("GOLDEN TENSOR= ", golden_tensor)

    for i in range(len(golden)):
        assert torch.isclose(golden_tensor[i],res_tensor[i], rtol = rtol, atol = atol), f"Failed at index {i} with values {golden[i]} and {res_from_L1[i]}"

    _ , pcc = comp_pcc(golden_tensor, res_tensor, pcc=0.99) 
    assert pcc > 0.99