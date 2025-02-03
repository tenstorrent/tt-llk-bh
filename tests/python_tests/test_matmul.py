import pytest
import torch
import os
from helpers import *

torch.set_printoptions(precision=2, linewidth=800, threshold=100000, sci_mode=False)

def generate_golden(operand1, operand2, data_format):
    # A_float = operand1.clone().detach()#.to(format_dict[data_format])
    # B_float = operand2.clone().detach()#.to(format_dict[data_format])

    A_untilized = untilize(operand1,data_format)
    B_untilized = untilize(operand2,data_format)

    result = torch.matmul(A_untilized, B_untilized )

    result = tilize(result)
    return result

formats = ["Float16_b", "Float16"]
@pytest.mark.parametrize("input_format", formats)
@pytest.mark.parametrize("output_format", formats)
@pytest.mark.parametrize("testname", ["matmul_test"])
@pytest.mark.parametrize("dest_acc", ["","DEST_ACC"])
def test_all(input_format, output_format, testname, dest_acc):
    if input_format != output_format:
        pytest.skip("")
    #context = init_debuda()
    src_A, src_B = generate_stimuli(input_format)
    golden_tensor = generate_golden(src_A, src_B, output_format)

    write_stimuli_to_l1(src_A, src_B, input_format)

    test_config = {
        "input_format": input_format,
        "output_format": output_format,
        "testname": testname,
        "dest_acc": dest_acc,
    }


    make_cmd = generate_make_command(test_config)
    os.system(f"cd .. && {make_cmd}")

    run_elf_files(testname)

    res_from_L1 = collect_results(output_format,src_A)

    os.system("cd .. && make clean")

    assert len(res_from_L1) == len(golden_tensor)

    # Mailbox checks
    assert read_words_from_device("0,0", 0x19FF4, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'
    assert read_words_from_device("0,0", 0x19FF8, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'
    assert read_words_from_device("0,0", 0x19FFC, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[output_format] if output_format in ["Float16", "Float16_b"] else torch.bfloat16)

    if(output_format == "Float16_b" or output_format == "Float16"):
        atol = 0.1
        rtol = 0.05
    elif(output_format == "Bfp8_b"):
        atol = 0.4
        rtol = 0.3

    rel_errs = []

    # for i in range(len(golden_tensor)):
    #     assert torch.isclose(golden_tensor[i],res_tensor[i], rtol = rtol, atol = atol), f"Failed at index {i} with values {golden_tensor[i]} and {res_from_L1[i]}"

    _ , pcc = comp_pcc(golden_tensor, res_tensor, pcc=0.99) 
    assert pcc > 0.98