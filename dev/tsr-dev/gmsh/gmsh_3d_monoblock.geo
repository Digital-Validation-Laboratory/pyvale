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
s1 = news; Rectangle(s1) = {-monoblock_width/2,0.0,0.0,monoblock_width,monoblock_width};
s8 = news; Rectangle(s8) = {-monoblock_width/2,monoblock_width,0.0,monoblock_width,monoblock_arm_height};
BooleanFragments{ Surface{s1}; Delete; }{ Surface{s8}; Delete; }

c2 = newc; Circle(c2) = {pipe_cent_x,pipe_cent_y,0.0,interlayer_rad_ext};
cl2 = newcl; Curve Loop(cl2) = {c2};
s2 = news; Plane Surface(s2) = {cl2};

c3 = newc; Circle(c3) = {pipe_cent_x,pipe_cent_y,0.0,pipe_rad_ext};
cl3 = newcl; Curve Loop(cl3) = {c3};
s3 = news; Plane Surface(s3) = {cl3};

c4 = newc; Circle(c4) = {pipe_cent_x,pipe_cent_y,0.0,pipe_rad_int};
cl4 = newcl; Curve Loop(cl4) = {c4};
s4 = news; Plane Surface(s4) = {cl4};

BooleanFragments{ Surface{s1}; Delete; }{ Surface{s2,s3}; Delete; }
BooleanDifference{ Surface{s3}; Delete; }{ Surface{s4}; Delete; }

Mesh.MeshSizeMin = elem_size;
Mesh.MeshSizeMax = elem_size;
Mesh 2;

