from .dictionaries import *

def generate_make_command(test_config):
    make_cmd = f"make --silent "

    input_format = test_config.get("input_format", "Float16_b") # Flolat16_b is default
    output_format = test_config.get("output_format", "Float16_b")
    testname = test_config.get("testname")
    dest_acc = test_config.get("dest_acc", " ") # default is not 32 bit dest_acc 

    make_cmd += f"format={format_args_dict[output_format]} testname={testname} dest_acc={dest_acc} " # jsut for now take output_format
    
    mathop = test_config.get("mathop", "no_mathop")

    if(mathop != "no_mathop"):
        make_cmd += f"mathop={mathop_args_dict[mathop]}"

    return make_cmd