import os
import re

def extract_functions(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 関数定義の正規表現パターン（C/C++向け）
    pattern = re.compile(r'^[\w\s\*]+?\s+(\w+)\s*\(([^)]*)\)\s*\{', re.MULTILINE)
    matches = pattern.finditer(content)

    functions = []
    for match in matches:
        func_name = match.group(1)
        args = match.group(2)
        functions.append({
            'function_name': func_name,
            'arguments': args,
            'file_path': file_path
        })

    return functions

# 使用例
functions = extract_functions('./ODrive/Firmware/MotorControl/controller.cpp')
for func in functions:
    print(func)
