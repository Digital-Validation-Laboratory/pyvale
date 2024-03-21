// Gmsh 3D parametric plate mesh with hole
// author: scepticalrabbit
SetFactory("OpenCASCADE");
General.Terminal = 1;

// View Options
Geometry.PointLabels = 0;
Geometry.CurveLabels = 0;
Geometry.SurfaceLabels = 1;
Geometry.VolumeLabels = 0;

// Variables
elem_size = 1e-3;
tol = elem_size/4;

plate_leng = 100e-3;
plate_height = 100e-3;
plate_thick = 10e-3;
hole_rad = 50e-3/2;
hole_loc_x = plate_leng/2;
hole_loc_y = plate_height/2;

hole_sect_nodes = 7;
plate_rad_nodes = 7;
plate_edge_nodes = Floor((hole_sect_nodes-1)/2)+1;
plate_thick_divs = 3;

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

// Transfinite line and surface meshing
Transfinite Curve{2,4,7,11} = plate_rad_nodes;
Transfinite Curve{1,5,14,15,12,13,8,9} = plate_edge_nodes;
Transfinite Curve{3,16,6,10} = hole_sect_nodes;

Transfinite Surface{s1} = {1,3,4,5};
Recombine Surface{s1};
Transfinite Surface{s2} = {4,5,7,6};
Recombine Surface{s2};
Transfinite Surface{s3} = {6,7,9,10};
Recombine Surface{s3};
Transfinite Surface{s4} = {3,9,10,1};
Recombine Surface{s4};

// Extrude the surface mesh to 3D
Extrude{0.0,0.0,plate_thick}{
    Surface{:}; Layers{plate_thick_divs}; Recombine;
}

// Meshing controls
Mesh.ElementOrder = 2;
Mesh 3;




