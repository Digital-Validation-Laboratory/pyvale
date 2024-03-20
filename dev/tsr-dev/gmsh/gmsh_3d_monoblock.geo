//------------------------------------------------------------------------------
// Gmsh: 3D monoblock
// author: scepticalrabbit
//------------------------------------------------------------------------------
SetFactory("OpenCASCADE");

// View Options
Geometry.PointLabels = 1;
Geometry.CurveLabels = 1;
Geometry.SurfaceLabels = 1;
Geometry.VolumeLabels = 1;
Geometry.Surfaces = 1;

//------------------------------------------------------------------------------
// Variables
elem_size = 1e-3;
tol = elem_size/4;
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

//------------------------------------------------------------------------------
// Geometry Definition

// Create a pattern of rectangles around the pipe hole to make mapped meshing
// easier
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

// Add all the blocks together and merge coincicent lines
BooleanFragments{ Surface{s1}; Delete; }{ Surface{s2,s3,s4,s5}; Delete; }

// Get the surfaces that will need to be hollowed out later for the pipe
pipe_block_surfs() = Surface In BoundingBox{-monoblock_width/2-tol,-tol,-tol,
    monoblock_width+tol,monoblock_width+tol,tol};

/*
For ss In {0:#pipe_block()-1}
    Printf("Surface in block: %g",pipe_block(ss));
EndFor
*/

// Generate circles for the pipe and the interlayer
c2 = newc; Circle(c2) = {pipe_cent_x,pipe_cent_y,0.0,interlayer_rad_ext};
cl2 = newcl; Curve Loop(cl2) = {c2};
s6 = news; Plane Surface(s6) = {cl2};

c3 = newc; Circle(c3) = {pipe_cent_x,pipe_cent_y,0.0,pipe_rad_ext};
cl3 = newcl; Curve Loop(cl3) = {c3};
s7 = news; Plane Surface(s7) = {cl3};

c4 = newc; Circle(c4) = {pipe_cent_x,pipe_cent_y,0.0,pipe_rad_int};
cl4 = newcl; Curve Loop(cl4) = {c4};
s8 = news; Plane Surface(s8) = {cl4};

// Remove the pipe hole and interlayer from the block
BooleanDifference{ Surface{pipe_block_surfs(),s6,s7}; Delete; }{ Surface{s8}; Delete;}

//------------------------------------------------------------------------------
// Transfinite Meshing
cn1() = Curve In BoundingBox{
    pipe_cent_x-pipe_rad_int-tol,pipe_cent_y-pipe_rad_int-tol,-tol,
    pipe_cent_x+pipe_rad_int+tol,pipe_cent_y+pipe_rad_int+tol,+tol};

For ss In {0:#cn1()-1}
    Printf("Curve: %g",cn1(ss));
EndFor
//sn1() = Surface In BoundingBox{};
//Transfinite Curve(cn1) = 3;
//Transfinite Surface(sn1);

//------------------------------------------------------------------------------
// Global Mesh controls
Mesh.MeshSizeMin = elem_size;
Mesh.MeshSizeMax = elem_size;
Mesh.ElementOrder = 2;
Mesh.HighOrderOptimize = 2;
//Mesh 2;

