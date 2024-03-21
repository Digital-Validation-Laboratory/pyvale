//------------------------------------------------------------------------------
// Gmsh: 3D monoblock
// author: scepticalrabbit
//------------------------------------------------------------------------------
SetFactory("OpenCASCADE");
General.Terminal = 1;

// Set to number of threads available


// View Options
Geometry.PointLabels = 0;
Geometry.CurveLabels = 0;
Geometry.SurfaceLabels = 0;
Geometry.VolumeLabels = 1;
Geometry.Surfaces = 1;

//------------------------------------------------------------------------------
// Variables

// Geometry variables
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

// Mesh variables
elem_size = 1e-3;
tol = elem_size/4;

mesh_ref = 1;

pipe_sect_nodes = Round(mesh_ref*11); // Must be odd
pipe_rad_nodes = Round(mesh_ref*7);
interlayer_rad_nodes = Round(mesh_ref*7);
monoblock_side_nodes = Round(mesh_ref*9);
monoblock_arm_nodes = Round(mesh_ref*11);
monoblock_depth_nodes = Round(mesh_ref*9);

monoblock_width_nodes = Floor((pipe_sect_nodes-1)/2)+1;

Printf("=====================================================================");
Printf("VARIABLES");
Printf("monoblock_width_nodes = %g",monoblock_width_nodes);
Printf("=====================================================================");


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
sa1 = news; Rectangle(sa1) =
    {-monoblock_width/2,monoblock_width,0.0,monoblock_width/2,monoblock_arm_height};
sa2 = news; Rectangle(sa2) =
    {0.0,monoblock_width,0.0,monoblock_width/2,monoblock_arm_height};

// Add all the blocks together and merge coincicent lines
BooleanFragments{ Surface{s1}; Delete; }{ Surface{s2,s3,s4,sa1,sa2}; Delete; }

// Get the surfaces that will need to be hollowed out later for the pipe
pipe_block_surfs() = Surface In BoundingBox{-monoblock_width/2-tol,-tol,-tol,
    monoblock_width+tol,monoblock_width+tol,tol};

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
BooleanDifference{Surface{pipe_block_surfs(),s6,s7}; Delete;}{ Surface{s8}; Delete;}

//------------------------------------------------------------------------------
// Transfinite Line/Curve Meshing

// Mesh the circular lines for the pipe and interlaye
cm1() = Curve In BoundingBox{
    pipe_cent_x-interlayer_rad_ext-tol,pipe_cent_y-interlayer_rad_ext-tol,-tol,
    pipe_cent_x+interlayer_rad_ext+tol,pipe_cent_y+interlayer_rad_ext+tol,+tol};

Printf("=====================================================================");
For ss In {0:#cm1()-1}
    Printf("Meshing curve: %g",cm1(ss));
    Transfinite Curve(cm1(ss)) = pipe_sect_nodes;
EndFor
Printf("=====================================================================");

// Mesh the radial lines for the pipe

cm1() = Curve In BoundingBox{
    pipe_cent_x+pipe_rad_int-tol,pipe_cent_y-tol,-tol,
    pipe_cent_x+pipe_rad_ext+tol,pipe_cent_y+tol,+tol};
Transfinite Curve(cm1(0)) = pipe_rad_nodes;

cm2() = Curve In BoundingBox{
    pipe_cent_x-pipe_rad_ext-tol,pipe_cent_y-tol,-tol,
    pipe_cent_x-pipe_rad_int+tol,pipe_cent_y+tol,+tol};
Transfinite Curve(cm2(0)) = pipe_rad_nodes;

cm3() = Curve In BoundingBox{
    pipe_cent_x-tol,pipe_cent_y-pipe_rad_ext-tol,-tol,
    pipe_cent_x+tol,pipe_cent_y-pipe_rad_int+tol,+tol};
Transfinite Curve(cm3(0)) = pipe_rad_nodes;

cm4() = Curve In BoundingBox{
    pipe_cent_x-tol,pipe_cent_y+pipe_rad_int-tol,-tol,
    pipe_cent_x+tol,pipe_cent_y+pipe_rad_ext+tol,+tol};
Transfinite Curve(cm4(0)) = pipe_rad_nodes;

// Mesh the radial lines for the interlayer
cm1() = Curve In BoundingBox{
    pipe_cent_x+interlayer_rad_int-tol,pipe_cent_y-tol,-tol,
    pipe_cent_x+interlayer_rad_ext+tol,pipe_cent_y+tol,+tol};
Transfinite Curve(cm1(0)) = interlayer_rad_nodes;

cm2() = Curve In BoundingBox{
    pipe_cent_x-interlayer_rad_ext-tol,pipe_cent_y-tol,-tol,
    pipe_cent_x-interlayer_rad_int+tol,pipe_cent_y+tol,+tol};
Transfinite Curve(cm2(0)) = interlayer_rad_nodes;

cm3() = Curve In BoundingBox{
    pipe_cent_x-tol,pipe_cent_y-interlayer_rad_ext-tol,-tol,
    pipe_cent_x+tol,pipe_cent_y-interlayer_rad_int+tol,+tol};
Transfinite Curve(cm3(0)) = interlayer_rad_nodes;

cm4() = Curve In BoundingBox{
    pipe_cent_x-tol,pipe_cent_y+interlayer_rad_int-tol,-tol,
    pipe_cent_x+tol,pipe_cent_y+interlayer_rad_ext+tol,+tol};
Transfinite Curve(cm4(0)) = interlayer_rad_nodes;

// Mesh the lines of the tungsten block joining the circular radial lines
cm1() = Curve In BoundingBox{
    pipe_cent_x+interlayer_rad_ext-tol,pipe_cent_y-tol,-tol,
    pipe_cent_x+monoblock_width/2+tol,pipe_cent_y+tol,+tol};
Transfinite Curve(cm1(0)) = monoblock_side_nodes;

cm2() = Curve In BoundingBox{
    pipe_cent_x-monoblock_width/2-tol,pipe_cent_y-tol,-tol,
    pipe_cent_x-interlayer_rad_ext+tol,pipe_cent_y+tol,+tol};
Transfinite Curve(cm2(0)) = monoblock_side_nodes;

cm3() = Curve In BoundingBox{
    pipe_cent_x-tol,pipe_cent_y-monoblock_width/2-tol,-tol,
    pipe_cent_x+tol,pipe_cent_y-interlayer_rad_ext+tol,+tol};
Transfinite Curve(cm3(0)) = monoblock_side_nodes;

cm4() = Curve In BoundingBox{
    pipe_cent_x-tol,pipe_cent_y+interlayer_rad_ext-tol,-tol,
    pipe_cent_x+tol,pipe_cent_y+monoblock_width/2+tol,+tol};
Transfinite Curve(cm4(0)) = monoblock_side_nodes;

// Mesh the armour vertical lines first
cm1() = Curve In BoundingBox{
    -monoblock_width/2-tol,monoblock_width-tol,-tol,
    monoblock_width/2+tol,monoblock_width+monoblock_arm_height+tol,+tol};

Printf("=====================================================================");
Printf("Meshing the vertical lines of the armour block");
For ss In {0:#cm1()-1}
    Printf("Meshing curve: %g",cm1(ss));
    Transfinite Curve(cm1(ss)) = monoblock_arm_nodes;
EndFor
Printf("=====================================================================");

// Mesh the side lines of the monoblock
cm1() = Curve In BoundingBox{
    -monoblock_width/2-tol,-tol,-tol,
    -monoblock_width/2+tol,monoblock_width+tol,+tol};

Printf("=====================================================================");
Printf("Meshing left side of the monoblock lines");
For ss In {0:#cm1()-1}
    Printf("Meshing curve: %g",cm1(ss));
    Transfinite Curve(cm1(ss)) = monoblock_width_nodes;
EndFor
Printf("=====================================================================");


cm2() = Curve In BoundingBox{
    monoblock_width/2-tol,-tol,-tol,
    monoblock_width/2+tol,monoblock_width+tol,+tol};

Printf("=====================================================================");
Printf("Meshing right side of the monoblock lines");
For ss In {0:#cm2()-1}
    Printf("Meshing curve: %g",cm2(ss));
    Transfinite Curve(cm2(ss)) = monoblock_width_nodes;
EndFor
Printf("=====================================================================");

// Mesh top,bottom and armour horizontal lines
cm3() = Curve In BoundingBox{
    -monoblock_width/2-tol,-tol,-tol,
    monoblock_width/2+tol,+tol,+tol};

Printf("=====================================================================");
Printf("Meshing bottom monoblock lines");
For ss In {0:#cm3()-1}
    Printf("Meshing curve: %g",cm3(ss));
    Transfinite Curve(cm3(ss)) = monoblock_width_nodes;
EndFor
Printf("=====================================================================");


cm3() = Curve In BoundingBox{
    -monoblock_width/2-tol,monoblock_width-tol,-tol,
    monoblock_width/2+tol,monoblock_width+tol,+tol};

Printf("=====================================================================");
Printf("Meshing top monoblock lines");
For ss In {0:#cm3()-1}
    Printf("Meshing curve: %g",cm3+monoblock_arm_height+(ss));
    Transfinite Curve(cm3(ss)) = monoblock_width_nodes;
EndFor
Printf("=====================================================================");


cm4() = Curve In BoundingBox{
    -monoblock_width/2-tol,monoblock_width+monoblock_arm_height+-tol,-tol,
    monoblock_width/2+tol,monoblock_width+monoblock_arm_height+tol,+tol};

Printf("=====================================================================");
Printf("Meshing top armour lines");
For ss In {0:#cm4()-1}
    Printf("Meshing curve: %g",cm4(ss));
    Transfinite Curve(cm4(ss)) = monoblock_width_nodes;
EndFor
Printf("=====================================================================");

//------------------------------------------------------------------------------
// Transfinite Surface Meshing

// Mesh the pipe and interlayer
sm1() = Surface In BoundingBox{
    pipe_cent_x-interlayer_rad_ext-tol,pipe_cent_y-interlayer_rad_ext-tol,-tol,
    pipe_cent_x+interlayer_rad_ext+tol,pipe_cent_y+interlayer_rad_ext+tol,+tol};

Printf("=====================================================================");
Printf("Surface meshing pipe and interlayer");
For ss In {0:#sm1()-1}
    Printf("Meshing surface: %g",sm1(ss));
    Transfinite Surface{sm1(ss)};
    Recombine Surface{sm1(ss)};
EndFor
Printf("=====================================================================");

// Mesh the armour
sm1() = Surface In BoundingBox{
    -monoblock_width/2-tol,monoblock_width-tol,-tol,
    monoblock_width/2+tol,monoblock_width+monoblock_arm_height+tol,+tol};

Printf("=====================================================================");
Printf("Surface meshing the armour");
For ss In {0:#sm1()-1}
    Printf("Meshing surface: %g",sm1(ss));
    Transfinite Surface{sm1(ss)};
    Recombine Surface{sm1(ss)};
EndFor
Printf("=====================================================================");

// Mesh the block around the interlayer
Transfinite Surface{22} = {15,16,17,13};
Recombine Surface{22};

Transfinite Surface{27} = {15,24,5,13};
Recombine Surface{27};

Transfinite Surface{30} = {24,5,28,27};
Recombine Surface{30};

Transfinite Surface{31} = {27,28,17,16};
Recombine Surface{31};

//------------------------------------------------------------------------------
// Extrude the surface mesh to 3D
Extrude{0.0,0.0,monoblock_depth}{
    Surface{:}; Layers{monoblock_depth_nodes}; Recombine;
}

//------------------------------------------------------------------------------
// Global Mesh controls
num_threads = 8;
General.NumThreads = num_threads;
Mesh.MaxNumThreads1D = num_threads;
Mesh.MaxNumThreads2D = num_threads;
Mesh.MaxNumThreads3D = num_threads;

Mesh.Algorithm = 6;
Mesh.Algorithm3D = 10;

// Mesh.MeshSizeMin = elem_size;
// Mesh.MeshSizeMax = elem_size;
Mesh.ElementOrder = 2;
Mesh 3;

//Exit;

