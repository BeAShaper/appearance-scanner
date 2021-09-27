Microsoft Windows [Version 6.1.7601]
Copyright (c) 2009 Microsoft Corporation.  All rights reserved.

C:\Users\zm>g:

G:\>cd G:\QQDownLoad\zm_uv

G:\QQDownLoad\zm_uv>atlas_map.exe
error : missing required input(s) <default input> /w /h

**(C) Hongzhi Wu, Oct. 2008**
Built on Jan  9 2009

atlas_map  input_mesh_name

/w       width
/h       height
/g       gutter_size
/o       output_map_name
/v       output visualization of the gutter map

/help    (Show help)


G:\QQDownLoad\zm_uv>atlas_map.exe shape_3.obj /w 1024 /h 1024 /g 3 /v

G:\QQDownLoad\zm_uv>uvatlas

UVAtlas - a command line tool for generating UV Atlases

Usage: UVAtlas.exe [options] [filename1] [filename2] ...

where:

  [/n #]        Specifies the maximum number of charts to generate
                Default is 0 meaning the atlas will be parameterized based
                solely on stretch
  [/st #.##]    Specifies the maximum amount of stretch, valid range is [0-1]
                Default is 0.16667. 0.0 means do not stretch; 1.0 means any
                amount of stretching is allowed.
  [/g #.##]     Specifies the gutter width (default 2).
  [/w #]        Specifies the texture width (default 512).
  [/h #]        Specifies the texture height (default 512).
  [/uvi #]      Specifies the output D3DDECLUSAGE_TEXCOORD index for the
                UVAtlas data (default 0).
  [/ta]         Generate topological adjacency, where triangles are marked
                adjacent if they share edge vertices. Mutually exclusive with
                /ga & /fa.
  [/ga]         Generate geometric adjacency, where triangles are marked
                adjacent if edge vertices are positioned within 1e-5 of each
                other. Mutually exclusive with /ta & /fa.
  [/fa file]    Load adjacency array entries directly into memory from
                a binary file. Mutually exclusive with /ta & /ga.
  [/fe file]    Load "False Edge" adjacency array entries directly into
                memory from a binary file. A non-false edge is indicated by -1,
                while a false edge is indicated by any other value, e.g. 0 or
                the original adjacency value. This enables the parameterization
                of meshes containing quads and higher order n-gons, and the
                internal edges of each n-gon will not be cut during the
                parameterization process.
  [/ip file]    Calculate the Integrated Metric Tensor (IMT) array for the mesh
                using a PRT buffer in file.
  [/it file]    Calculate the IMT for the mesh using a texture map in file.
  [/iv usage]   Calculate the IMT for the mesh using a per-vertex data from the
                mesh. The usage parameter lets you select which part of the
                mesh to use (default COLOR). It must be one of NORMAL, COLOR,
                TEXCOORD, TANGENT, or BINORMAL.
  [/t]          Create a separate mesh in u-v space (appending _texture).
  [/c]          Modify the materials of the mesh to graphically show
                which chart each triangle is in.
  [/rt file]    Resamples a texture using the new UVAtlas parameterization.
                The resampled texture is saved to a filename with "_resampled"
                appended. Defaults to reading old texture parameterization from
                D3DDECLUSAGE_TEXCOORD[0] in original mesh Use /rtu and /rti to
                override this.
  [/rtu usage]  Specifies the vertex data usage for texture resampling (default
                TEXCOORD). It must be one of NORMAL, POSITION, COLOR, TEXCOORD,
                TANGENT, or BINORMAL.
  [/rti #]      Specifies the usage index for texture resampling (default 0).
  [/o file]     Output mesh filename.  Defaults to a filename with "_result"
                appended Using this option disables batch processing.
  [/f]          Overwrite original file with output (default off).
                Mutually exclusive with /o.
  [/q usage]    Quality flag for D3DXUVAtlasCreate. It must be
                either DEFAULT, FAST, or QUALITY.
  [/s]          Search sub-directories for files (default off).
  [filename*]   Specifies the files to generate atlases for.
                Wildcards and quotes are supported.


G:\QQDownLoad\zm_uv>conv_mesh
error : missing required input(s) <default input> /o

**(C) Hongzhi Wu, Oct. 2008**
Built on Oct 30 2008

conv_mesh  input_mesh_file_name

/o       output_mesh_file_name

/help    (Show help)


G:\QQDownLoad\zm_uv>conv_mesh shape_3.obj /o shape_3.x

G:\QQDownLoad\zm_uv>uvatlas

UVAtlas - a command line tool for generating UV Atlases

Usage: UVAtlas.exe [options] [filename1] [filename2] ...

where:

  [/n #]        Specifies the maximum number of charts to generate
                Default is 0 meaning the atlas will be parameterized based
                solely on stretch
  [/st #.##]    Specifies the maximum amount of stretch, valid range is [0-1]
                Default is 0.16667. 0.0 means do not stretch; 1.0 means any
                amount of stretching is allowed.
  [/g #.##]     Specifies the gutter width (default 2).
  [/w #]        Specifies the texture width (default 512).
  [/h #]        Specifies the texture height (default 512).
  [/uvi #]      Specifies the output D3DDECLUSAGE_TEXCOORD index for the
                UVAtlas data (default 0).
  [/ta]         Generate topological adjacency, where triangles are marked
                adjacent if they share edge vertices. Mutually exclusive with
                /ga & /fa.
  [/ga]         Generate geometric adjacency, where triangles are marked
                adjacent if edge vertices are positioned within 1e-5 of each
                other. Mutually exclusive with /ta & /fa.
  [/fa file]    Load adjacency array entries directly into memory from
                a binary file. Mutually exclusive with /ta & /ga.
  [/fe file]    Load "False Edge" adjacency array entries directly into
                memory from a binary file. A non-false edge is indicated by -1,
                while a false edge is indicated by any other value, e.g. 0 or
                the original adjacency value. This enables the parameterization
                of meshes containing quads and higher order n-gons, and the
                internal edges of each n-gon will not be cut during the
                parameterization process.
  [/ip file]    Calculate the Integrated Metric Tensor (IMT) array for the mesh
                using a PRT buffer in file.
  [/it file]    Calculate the IMT for the mesh using a texture map in file.
  [/iv usage]   Calculate the IMT for the mesh using a per-vertex data from the
                mesh. The usage parameter lets you select which part of the
                mesh to use (default COLOR). It must be one of NORMAL, COLOR,
                TEXCOORD, TANGENT, or BINORMAL.
  [/t]          Create a separate mesh in u-v space (appending _texture).
  [/c]          Modify the materials of the mesh to graphically show
                which chart each triangle is in.
  [/rt file]    Resamples a texture using the new UVAtlas parameterization.
                The resampled texture is saved to a filename with "_resampled"
                appended. Defaults to reading old texture parameterization from
                D3DDECLUSAGE_TEXCOORD[0] in original mesh Use /rtu and /rti to
                override this.
  [/rtu usage]  Specifies the vertex data usage for texture resampling (default
                TEXCOORD). It must be one of NORMAL, POSITION, COLOR, TEXCOORD,
                TANGENT, or BINORMAL.
  [/rti #]      Specifies the usage index for texture resampling (default 0).
  [/o file]     Output mesh filename.  Defaults to a filename with "_result"
                appended Using this option disables batch processing.
  [/f]          Overwrite original file with output (default off).
                Mutually exclusive with /o.
  [/q usage]    Quality flag for D3DXUVAtlasCreate. It must be
                either DEFAULT, FAST, or QUALITY.
  [/s]          Search sub-directories for files (default off).
  [filename*]   Specifies the files to generate atlases for.
                Wildcards and quotes are supported.


G:\QQDownLoad\zm_uv>uvatlas /g 3 /w 512 /h 512 shape_3.x
Searching dir G:\QQDownLoad\zm_uv\ for shape_3.x
Processing file G:\QQDownLoad\zm_uv\shape_3.x
Face count: 1622
Vertex count: 874
Max charts: Atlas will be parameterized based solely on stretch
Max stretch: 0.166667
Texture size: 512 x 512
Quality: D3DXUVATLAS_DEFAULT
Gutter size: 3.000000 texels
Updating UVs in mesh's D3DDECLUSAGE_TEXCOORD[0]
Executing D3DXUVAtlasCreate() on mesh...
D3DXUVAtlasCreate() succeeded
Output # of charts: 3
Output stretch: 0.077420
Output mesh with new UV atlas: G:\QQDownLoad\zm_uv\shape_3_result.x


G:\QQDownLoad\zm_uv>conv_mesh shape_3_result.x /o shape_3_result.obj

G:\QQDownLoad\zm_uv>atlas_map.exe shape_3_result.obj /w 512 /h 512 /g 3 /v

G:\QQDownLoad\zm_uv>