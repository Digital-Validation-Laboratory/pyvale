// Gmsh 2D parametric plate mesh
// author: scepticalrabbit
SetFactory("OpenCASCADE");

// View Options
Geometry.PointLabels = 1;
Geometry.CurveLabels = 1;
Geometry.SurfaceLabels = 1;
Geometry.VolumeLabels = 1;

// Variables
elem_size = 0.5e-3;
plate_leng = 100e-3;
plate_height = 100e-3;
hole_rad = 50e-3/2;
hole_loc_x = plate_leng/2;
hole_loc_y = plate_height/2;

// Geometry Definition
s1 = news; Rectangle(s1) = {0.0,0.0,0.0,plate_leng/2,plate_height/2};
s2 = news; Rectangle(s2) = {plate_leng/2,0.0,0.0,plate_leng/2,plate_height/2};
s3 = news; Rectangle(s3) = {plate_leng/2,plate_height/2,0.0,plate_leng/2,plate_height/2};
s4 = news; Rectangle(s4) = {0.0,plate_height/2,0.0,plate_leng/2,plate_height/2};
BooleanFragments{ Surface{s1}; Delete; }{ Surface{s2,s3,s4}; Delete; }

c2 = newc; Circle(c2) = {hole_loc_x,hole_loc_y,0.0,hole_rad};
cl2 = newcl; Curve Loop(cl2) = {c2};
s5 = news; Plane Surface(s5) = {cl2};

BooleanDifference{ Surface{s1,s2,s3,s4}; Delete; }{ Surface{s5}; Delete; }

Transfinite Curve{2,4,7,11} = 5;
Transfinite Curve{1,5} = 5;
Transfinite Curve{3} = 9;

Transfinite Surface{s1} = {1,3,4,5};
Recombine Surface{s1};


//Mesh.MeshSizeMin = elem_size;
//Mesh.MeshSizeMax = elem_size;

Mesh.ElementOrder = 2;
Mesh.HighOrderOptimize = 2;
Mesh 2;


