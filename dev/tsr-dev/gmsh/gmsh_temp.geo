// Gmsh: 3D monoblock
// author: scepticalrabbit
SetFactory("OpenCASCADE");

// View Options
Geometry.PointLabels = 1;
Geometry.CurveLabels = 1;
Geometry.SurfaceLabels = 1;
Geometry.VolumeLabels = 1;
Geometry.Surfaces = 1;

// Variables
elem_size = 0.5e-3;

pipe_rad_int = 6e-3;
pipe_thick = 1.5e-3;
pipe_rad_ext = pipe_rad_int+pipe_thick;

interlayer_thick = 2e-3;
interlayer_rad_int = pipe_rad_ext;
interlayer_rad_ext = interlayer_rad_int+interlayer_thick;

monoblock_depth = 12e-3;
monoblock_side = 3e-3;
monoblock_arm_height = 8e-3;
monoblock_width = 2*interlayer_rad_ext + 2*monoblock_side;
monoblock_height = monoblock_width + monoblock_arm_height;

pipe_cent_x = 0.0;
pipe_cent_y = interlayer_rad_ext + monoblock_side;

// Geometry
s1 = news; Rectangle(s1) =
    {-monoblock_width/2,0.0,0.0,monoblock_width/2,monoblock_width/2};
s2 = news; Rectangle(s2) =
    {-monoblock_width/2,monoblock_width/2,0.0,monoblock_width/2,monoblock_width/2};
s3 = news; Rectangle(s3) =
    {0.0,monoblock_width/2,0.0,monoblock_width/2,monoblock_width/2};
s4 = news; Rectangle(s4) =
    {0.0,0.0,0.0,monoblock_width/2,monoblock_width/2};
s5 = news; Rectangle(s5) =
    {-monoblock_width/2,monoblock_width,0.0,monoblock_width,monoblock_arm_height};
block_surfs() = BooleanFragments{ Surface{s1}; Delete; }{ Surface{s2,s3,s4,s5}; Delete; };

For ss In {0:#block_surfs()-1}
    Printf("Surface in block: %g",block_surfs(ss));
EndFor
