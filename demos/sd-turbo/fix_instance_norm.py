import os
import re
import sys


assert len(sys.argv) == 3, 'Syntax: python fix_instance_norm.py INPUT_TXT_OR_ONNX OUTPUT_TXT_OR_ONNX'
input_file = sys.argv[1]
output_file = sys.argv[2]

if os.path.splitext(input_file)[-1] == '.onnx':
    converted_input_file = input_file + '.txt'
    os.system(f'onnx2text "{input_file}" "{converted_input_file}"')
    input_file = converted_input_file

converted_output_file = None
if os.path.splitext(output_file)[-1] == '.onnx':
    converted_output_file = output_file
    output_file += '.txt'

with open(input_file, 'r', encoding='utf8') as f:
    text = f.read()

replace = '''node {
    input: "\\2\\3"
    output: "\\2\\3_fromcast"
    name: "\\2_fromcast"
    op_type: "Cast"
    attribute {
      name: "to"
      i: 1
      type: INT
    }
  }
  \\1\\2\\3_fromcast\\4\\5\\6_tocast\\7node {
    input: "\\5\\6_tocast"
    output: "\\5\\6"
    name: "\\5_tocast"
    op_type: "Cast"
    attribute {
      name: "to"
      i: 10
      type: INT
    }
  }  
  '''

delim = '(?=\\bnode {)'
parts = re.split(delim, text)
total = 0
constants = []
for i, part in enumerate(parts[1 : -1], start=1):
    part, subs = re.subn('(.*?input: ")([^"]*?)(_output[^"]*)?(".*?output: ")([^"]*?)(_output[^"]*)?(".*op_type: "InstanceNormalization".*)', replace, part, flags=re.DOTALL)
    if subs:
        constants += re.findall('".*constant.*"', part, flags=re.IGNORECASE)
        parts[i] = part
        total += 1
for i, part in enumerate(parts[1 : -1], start=1):
    if any(f'output: {const}' in part for const in constants):
        parts[i] = part.replace('data_type: 10', 'data_type: 1'
                    ).replace(
                        r'raw_data: "\000<\000<\000<\000<\000<\000<\000<\000<\000<\000<\000<\000<\000<\000<\000<\000<\000<\000<\000<\000<\000<\000<\000<\000<\000<\000<\000<\000<\000<\000<\000<\000<"',
                        r'raw_data: "\000\000\200?\000\000\200?\000\000\200?\000\000\200?\000\000\200?\000\000\200?\000\000\200?\000\000\200?\000\000\200?\000\000\200?\000\000\200?\000\000\200?\000\000\200?\000\000\200?\000\000\200?\000\000\200?\000\000\200?\000\000\200?\000\000\200?\000\000\200?\000\000\200?\000\000\200?\000\000\200?\000\000\200?\000\000\200?\000\000\200?\000\000\200?\000\000\200?\000\000\200?\000\000\200?\000\000\200?\000\000\200?"'
                    ).replace(
                        r'raw_data: "\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000"',
                        r'raw_data: "\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000"'
                    )
print(f'Fixed {total} InstanceNormalization nodes')
with open(output_file, 'w', encoding='utf8') as f:
    f.write(''.join(parts))

if converted_output_file:
    os.system(f'onnx2text "{output_file}" "{converted_output_file}"')
