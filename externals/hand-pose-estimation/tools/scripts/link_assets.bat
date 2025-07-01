@echo off
SETLOCAL EnableDelayedExpansion

:: relative path from repository root to build directories
set "release_path=build\x64_Release"
set "debug_path=build\x64_Debug"

cd "%~dp0"
cd "..\.."
echo %cd%

echo.SETUP folders
set "rel_assets=%cd%\build\x64_Release\assets"
if not exist %rel_assets% mkdir %rel_assets%
set "dbg_assets=%cd%\build\x64_Debug\assets"
if not exist %dbg_assets% mkdir %dbg_assets%

if exist "%cd%\assets\shaders" (
	xcopy "%cd%\assets\shaders\*" %release_path% /I /C /Q /Y
	xcopy "%cd%\assets\shaders\*" %debug_path% /I /C /Q /Y
)

echo.LINK assets recursively
call:link_asset_subfolders
call:iterate_externals
goto:eof

:link_asset_subfolders
if exist "assets" (
	for /d %%i in ("!cd!\assets\*") do ( 
		if not exist "!rel_assets!\%%~nxi" mklink /J "!rel_assets!\%%~nxi" "%%i"
		if not exist "!dbg_assets!\%%~nxi" mklink /J "!dbg_assets!\%%~nxi" "%%i"	
	)
)
goto:eof

:iterate_externals
if exist "externals" (
	for /d %%i in ("!cd!\externals\*") do ( 
		cd "%%i"
		call:link_asset_subfolders
		call:iterate_externals
		cd "..\.."
	)
)
goto:eof