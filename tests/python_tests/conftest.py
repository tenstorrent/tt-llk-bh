import pytest
import csv
from tabulate import tabulate

@pytest.fixture(scope='session')
def test_results():
    # results = []
    # yield results  # This makes the fixture available to tests
    
    # pass_results = [entry for entry in results if entry[0] == "PASS"]
    # fail_results = [entry for entry in results if entry[0] == "FAIL"]

    # # After all tests are done, we'll organize and print the collected info
    # headers = ["RESULT", "INPUT", "OUTPUT", "Unpack Src", "Unpack Dst", "FPU", "Pack Src", "Pack Dst", "Math", "Dest Acc"]
    
    # def sort_key(entry):
    #     # Sorting by mathop first, then by dest_acc ("" for off comes before "DEST_ACC" for on)
    #     return (entry[1], entry[2], entry[8], entry[8])  
    
    # sorted_pass_results = sorted(pass_results, key=sort_key)
    # sorted_fail_results = sorted(fail_results, key=sort_key)
    
    # with open('test_results.txt', 'w') as file:
    #     file.write("\n--- Collected Test Info (PASS) ---\n")
    #     file.write(tabulate(sorted_pass_results, headers=headers, tablefmt="grid"))
    #     file.write("\n\n--- Collected Test Info (FAIL) ---\n")
    #     file.write(tabulate(sorted_fail_results, headers=headers, tablefmt="grid"))
    
    
    results = []
    yield results  # This makes the fixture available to tests
    
    pass_results  = [] #= [entry for entry in results if entry[0] == "PASS"]
    fail_results = [] #[entry for entry in results if entry[0] == "FAIL"]
    for entry in results:
        if entry[0] == "PASS":
            pass_results.append(entry[0:4] + entry[5:])
        else:
            fail_results.append(entry[0:3] + entry[4:])

    # After all tests are done, we'll organize and print the collected info
    # headers = ["RESULT", "INPUT", "OUTPUT", "Unpack Src", "Unpack Dst", "FPU", "Pack Src", "Pack Dst", "Math", "Dest Acc"]
    headers_pass = ["RESULT", "INPUT", "OUTPUT", "PCC", "Unpack Src", "Unpack Dst", "FPU", "Pack Src", "Pack Dst", "Math", "Dest Acc"]
    headers_fail = ["RESULT", "INPUT", "OUTPUT", "ERROR", "Unpack Src", "Unpack Dst", "FPU", "Pack Src", "Pack Dst", "Math", "Dest Acc"]
    
    def sort_key(entry):
        # Sorting by mathop first, then by dest_acc ("" for off comes before "DEST_ACC" for on)
        return (entry[1], entry[2], entry[9], entry[8], -entry[3])  
    
    sorted_pass_results = sorted(pass_results, key=sort_key)
    sorted_fail_results = sorted(fail_results, key=sort_key)
    
    with open('test_results.txt', 'w') as file:
        file.write("\n--- Collected Test Info (PASS) ---\n")
        file.write(tabulate(sorted_pass_results, headers=headers_pass, tablefmt="grid"))
        file.write("\n\n--- Collected Test Info (FAIL) ---\n")
        file.write(tabulate(sorted_fail_results, headers=headers_fail, tablefmt="grid"))
        
    with open('test_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the headers for the pass results
        writer.writerow(headers_pass)
        
        # Write the data for pass results
        writer.writerows(sorted_pass_results)
        
        # Add a separator between pass and fail results
        writer.writerow([])  # Empty row
        
        # Write the headers for the fail results
        writer.writerow(headers_fail)
        
        # Write the data for fail results
        writer.writerows(sorted_fail_results)