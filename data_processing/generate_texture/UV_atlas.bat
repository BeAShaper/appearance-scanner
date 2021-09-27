SET UV_ATLAS_PATH=../uv/
cd /d %UV_ATLAS_PATH%
@echo on
uv_generator.bat %DATA_ROOT%%UDT_FOLDER_NAME%/ %~dp0
@echo off
