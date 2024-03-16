SetFactory("OpenCASCADE");

Point(1) = {0, 0, 0, 1.0};
Point(2) = {2, 0, 0, 1.0};
Point(3) = {2, 1, 0, 1.0};
Point(4) = {0, 1, 0, 1.0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Circle(5) = {1, 0.5, 0, 0.25, 0, 2*Pi};

Curve Loop(1) = {3, 4, 1, 2};
Curve Loop(2) = {5};
Plane Surface(1) = {1, 2};

Physical Surface("plate") = {1};

Mesh.size = 0.01;

Mesh 2;
