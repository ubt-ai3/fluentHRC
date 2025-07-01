import sys
import os.path
import shutil

if len(sys.argv) < 3:
    print("Arguments <runtime_deps_file> <output_dir>")
    raise Exception()

st_path = sys.argv[1]
out_path = sys.argv[2]

if not os.path.isfile(st_path):
    print("Invalid input file")
    raise Exception()

if not os.path.isdir(out_path):
    print("Invalid output dir")
    raise Exception()

content = None
with open(st_path, "r") as f:
    content = f.read()

alreay_failed = False

lines = content.split(";")
for file in lines:
    if (not os.path.isfile(file)):
        print("File: " + file + " does not exist")
        alreay_failed = True
        
if alreay_failed:
    raise Exception("File not found")

for file in lines:
    shutil.copy2(file, out_path)