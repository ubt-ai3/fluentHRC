@REM Copy assets folder to build folders for all configurations
@REM and all platforms of some project ($1).
@REM Attention: Project names with spaces are not supported.

@echo Copying new and modified assets to build directories for: %1

@cd /d "%~dp0"

@call util_copy_asset_folder.bat ^
	"..\..\assets\%1" ^
	"..\..\build\x64_Debug\%1" ^
	"..\..\build\temp\x64_Debug\%1"
	
@call util_copy_asset_folder.bat ^
	"..\..\assets\%1" ^
	"..\..\build\x64_Release\%1" ^
	"..\..\build\temp\x64_Release\%1"