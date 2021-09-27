@echo off
echo Start UVAtlas.

SET ROOT=%1
SET ORIGIN_FILE_NAME=meshed-poisson_obj_remeshed.obj
SET SAVE_FILE_NAME=meshed-poisson_obj_result
SET SAVE_FILE_NAME_OBJ=%SAVE_FILE_NAME%.obj
SET SAVE_FILE_NAME_ATLAS=%SAVE_FILE_NAME%.atlas_map
SET SAVE_FILE_NAME_VIS=%SAVE_FILE_NAME%_vis.png
SET TMP_DIR=%ROOT%tmp/
SET UVMAP_WIDTH=1024
SET UVMAP_HEIGHT=1024

md "%TMP_DIR%"

conv_mesh.exe %ROOT%%ORIGIN_FILE_NAME% /o %TMP_DIR%point-meshed.x
UVAtlas.exe /g 3 /w %UVMAP_WIDTH% /h %UVMAP_HEIGHT% %TMP_DIR%point-meshed.x
conv_mesh.exe %TMP_DIR%point-meshed_result.x /o %ROOT%%SAVE_FILE_NAME_OBJ%
atlas_map.exe %ROOT%%SAVE_FILE_NAME_OBJ% /g 3 /w %UVMAP_WIDTH% /h %UVMAP_HEIGHT% /v

SET MESH_DIR=%ROOT%mesh_%UVMAP_WIDTH%/

md "%MESH_DIR%"

SET DIR_BACK=%cd%
cd %ROOT%
move "%SAVE_FILE_NAME_OBJ%" "%MESH_DIR%"
move "%SAVE_FILE_NAME_ATLAS%" "%MESH_DIR%"
move "%SAVE_FILE_NAME_VIS%" "%MESH_DIR%"
del /Q tmp\*.*
rd tmp
cd /d %2
::%DIR_BACK%


echo UVAtlas DONE.