// Gmsh project created on Mon Jul 29 13:21:20 2024
SetFactory("OpenCASCADE");
//==============================================================================
// Gmsh 3D Monoblock
// author: Lloyd Fletcher (scepticalrabbit)
//==============================================================================

// View options: turns on/off which pieces of geometry we show
Geometry.Points = 1;
Geometry.Curves = 1;
Geometry.Surfaces = 1;
Geometry.Volumes = 1;

// View options: turns labels on and off for different types of geometry
Geometry.PointLabels = 1;
Geometry.CurveLabels = 0;
Geometry.SurfaceLabels = 1;
Geometry.VolumeLabels = 0;

//------------------------------------------------------------------------------
// Variable Definitions

//------------------------------------------------------------------------------
//_* MOOSEHERDER VARIABLES - START
file_name = "case11.msh";
num_threads = 7;

// Specified Geometry variables
pipe_rad_int = 6e-3;
pipe_thick = 1.5e-3;

interlayer_thick = 2e-3;

monoblock_depth = 12e-3;
monoblock_side = 3e-3;
monoblock_arm_height = 8e-3;

// Specified Mesh variables
base_divs = 1;
mesh_ref = 1; //  Must be an integer greater than 0

//_* MOOSEHERDER VARIABLES - END
//------------------------------------------------------------------------------

// Calculated Geometry Variables
pipe_rad_ext = pipe_rad_int+pipe_thick;

interlayer_rad_int = pipe_rad_ext;
interlayer_rad_ext = interlayer_rad_int+interlayer_thick;

monoblock_width = 2*interlayer_rad_ext + 2*monoblock_side;
monoblock_height = monoblock_width + monoblock_arm_height;

pipe_cent_x = 0.0;
pipe_cent_y = interlayer_rad_ext + monoblock_side;

// Calculated Mesh Variables
/*
pipe_sect_nodes = Round(mesh_ref*5); // Must be odd
pipe_rad_nodes = Round(mesh_ref*5);
interlayer_rad_nodes = Round(mesh_ref*5);
monoblock_side_nodes = Round(mesh_ref*5);
monoblock_arm_nodes = Round(mesh_ref*5);
monoblock_depth_nodes = Round(mesh_ref*2);
monoblock_width_nodes = Floor((pipe_sect_nodes-1)/2)+1;
*/

// This is a more reasonable mesh refinement for the monoblock but solve time
// is much longer
pipe_sect_nodes = Round(mesh_ref*11); // Must be odd
pipe_rad_nodes = Round(mesh_ref*7);
interlayer_rad_nodes = Round(mesh_ref*7);
monoblock_side_nodes = Round(mesh_ref*9);
monoblock_arm_nodes = Round(mesh_ref*11);
monoblock_depth_nodes = Round(mesh_ref*5);
monoblock_width_nodes = Floor((pipe_sect_nodes-1)/2)+1;


// Calculate approx element size by dividing the circumference
elem_size = 2*Pi*pipe_rad_int/(4*(pipe_sect_nodes-1));
tol = elem_size/4; // Used for selection tolerance of bounding boxes

//------------------------------------------------------------------------------
// Geometry Definition
s1 = news;
Rectangle(s1) =
    {-monoblock_width/2,0.0,0.0,
    monoblock_width/2,monoblock_width/2};

s2 = news;
Rectangle(s2) =
    {-monoblock_width/2,monoblock_width/2,0.0,
    monoblock_width/2,monoblock_width/2};

s3 = news;
Rectangle(s3) =
    {0.0,monoblock_width/2,0.0,
    monoblock_width/2,monoblock_width/2};

s4 = news;
Rectangle(s4) =
    {0.0,0.0,0.0,
    monoblock_width/2,monoblock_width/2};

sa1 = news;
Rectangle(sa1) =
    {-monoblock_width/2,monoblock_width,0.0,
    monoblock_width/2,monoblock_arm_height};

sa2 = news;
Rectangle(sa2) =
    {0.0,monoblock_width,0.0,
    monoblock_width/2,monoblock_arm_height};

BooleanFragments{ Surface{s1}; Delete; }{ Surface{s2,s3,s4,sa1,sa2}; Delete; }

pipe_block_surfs() = Surface In BoundingBox{
        -monoblock_width/2-tol,-tol,-tol,
        monoblock_width+tol,monoblock_width+tol,tol};

c2 = newc; Circle(c2) = {pipe_cent_x,pipe_cent_y,0.0,interlayer_rad_ext};
cl2 = newcl; Curve Loop(cl2) = {c2};
s6 = news; Plane Surface(s6) = {cl2};

c3 = newc; Circle(c3) = {pipe_cent_x,pipe_cent_y,0.0,pipe_rad_ext};
cl3 = newcl; Curve Loop(cl3) = {c3};
s7 = news; Plane Surface(s7) = {cl3};

c4 = newc; Circle(c4) = {pipe_cent_x,pipe_cent_y,0.0,pipe_rad_int};
cl4 = newcl; Curve Loop(cl4) = {c4};
s8 = news; Plane Surface(s8) = {cl4};

BooleanDifference{Surface{pipe_block_surfs(),s6,s7}; Delete;}
                 { Surface{s8}; Delete;}
