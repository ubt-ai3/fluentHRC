import os
import sys
from pathlib import Path
from hashlib import md5
import tempfile
import subprocess
from shutil import move

def md5_file(file):
    with open(file, "r") as handle:
        return md5(handle.read().encode()).hexdigest()

def equal_file(file0, file1):
    return md5_file(file0) == md5_file(file1)

def intersection(lst1, lst2):
    return [value for value in lst1 if value in lst2]

def left_diff(lst1, lst2):
    return [value for value in lst1 if value not in lst2]

def right_diff(lst1, lst2):
    return left_diff(lst2, lst1)

script_dir = Path(os.path.realpath(sys.path[0]))

def find_tool_dir(script_dir):
    temp_dir = (script_dir / ".." / ".." / "externals/vcpkg/x64-windows/tools").resolve()
    if os.path.isfile(temp_dir / "grpc/grpc_cpp_plugin.exe") == True:
        return temp_dir

    #search for alternative tools dir
    temp_dir = (script_dir / ".." / ".." / "externals").resolve()
    for root, dirs, files in os.walk(temp_dir):
        for name in files:
            if name == "grpc_cpp_plugin.exe":
                return (Path(root) / "..").resolve() 

tools = find_tool_dir(script_dir)

plugin = tools / "grpc/grpc_cpp_plugin.exe"
protoc = tools / "protobuf/protoc.exe"

proto_dir = script_dir / "proto"
output_dir = script_dir / "generated" / "x64"

existing_files = list()

if os.path.isdir(output_dir):
    existing_files = os.listdir(output_dir)
else:
    os.makedirs(output_dir, exist_ok=True)


with tempfile.TemporaryDirectory() as tmpdirname:
    #print('created temporary directory', tmpdirname)
    tempdir = Path(tmpdirname)
    cmd = protoc.as_posix() + " -I " + proto_dir.as_posix() + " --grpc_out=" + tempdir.as_posix() + " --cpp_out=" + tempdir.as_posix() + " --plugin=protoc-gen-grpc=" + plugin.as_posix() + " " + proto_dir.as_posix() + "/*"
    #print(cmd)
    print(cmd)
    process = subprocess.Popen(cmd) #TODO:: no errors or messages are put out?
    outs, errs = process.communicate()

    if outs:
        print("output")
        print(outs)
    if errs:
        print("errs")
        print(errs)

    gen_files = os.listdir(tempdir)

    duplicated_files = intersection(existing_files, gen_files)
    to_delete_files = left_diff(existing_files, gen_files)
    to_copy_files = right_diff(existing_files, gen_files)

    for file in duplicated_files:
        if not equal_file(output_dir / file, tempdir / file):
            to_delete_files.append(file)
            to_copy_files.append(file)

    for file in to_delete_files:
        os.remove(output_dir / file)
        #print(output_dir / file)

    #print()
    for file in to_copy_files:
        move(Path(tempdir / file), Path(output_dir / file))
        #Path(tempdir / file).rename(output_dir / file)
        #print(tempdir / file, " -> ", output_dir / file)
