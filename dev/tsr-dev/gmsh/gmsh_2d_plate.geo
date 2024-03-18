// Gmsh 2D parametric plate mesh
// author: scepticalrabbit
SetFactory("OpenCASCADE");

// View Options
Geometry.PointLabels = 1;
Geometry.CurveLabels = 1;
Geometry.SurfaceLabels = 1;
Geometry.VolumeLabels = 1;

// Variables
elem_size = 1e-3;
plate_leng = 100e-3;
plate_height = 50e-3;
hole_rad = 25e-3/2;
hole_loc_x = plate_leng/2;
hole_loc_y = plate_height/2;

// Geometry Definition
// Point: {x,y,z,targ_elem_size}
Point(1) = {0,0,0,elem_size};
Point(2) = {plate_leng,0,0,elem_size};
Point(3) = {plate_leng,plate_height,0,elem_size};
Point(4) = {0,plate_height,0,elem_size};

// Curves: Lines to create rectangle
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};

// Curves: Circle - must increment from line number as also a curve!
// OpenCASCADE: {CentX,CentY,CentZ,Rad,StartArcAng,EndArcAng}
Circle(5) = {hole_loc_x, hole_loc_y, 0.0, hole_rad,0.0,2*Pi};
//Circle(6) = {hole_loc_x, hole_loc_y, 0.0, hole_rad,Pi/2,Pi};
//Circle(7) = {hole_loc_x, hole_loc_y, 0.0, hole_rad,Pi,3*Pi/2};
//Circle(8) = {hole_loc_x, hole_loc_y, 0.0, hole_rad,3*Pi/2,2*Pi};


Transfinite Curve{5} = 21;

// Create loops from the lines
Curve Loop(1) = {1,2,3,4};
Curve Loop(2) = {5};

// Create surface from loops
// First must be the outer loop - remainder are inner loops or holes
Plane Surface(1) = {1,2};

Physical Surface("plate") = {1};

//Mesh.MeshSizeMin = elem_size;
//Mesh.MeshSizeMax = elem_size;
Mesh 2;

/*
Plane Surface(1) = {1};

// Used for BCs
Physical Curve(5) = {1,2,4};
Physical Surface("My Surface") = {1};

// Mesh
Mesh 2;
Save "t1.msh";

// Exit;
*/
