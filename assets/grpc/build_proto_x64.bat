@echo off
set tools=%~dp0/../../externals/vcpkg/x64-windows/tools
set plugin=%tools%/grpc/grpc_cpp_plugin.exe
set protoc=%tools%/protobuf/protoc.exe

rmdir /s /q "%~dp0generated\x64"
mkdir "%~dp0\generated\x64"

@echo on
start /B /wait %protoc% -I %~dp0/proto --grpc_out=%~dp0/generated/x64 --cpp_out=%~dp0/generated/x64 --plugin=protoc-gen-grpc=%plugin% %~dp0/proto/*