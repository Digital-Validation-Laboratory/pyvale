SetFactory("OpenCASCADE");

// View Options
Geometry.PointLabels = 1;
Geometry.CurveLabels = 1;
Geometry.SurfaceLabels = 1;
Geometry.VolumeLabels = 1;

// Create the first rectangle using OpenCASCADE
Rectangle(1) = {0, 0, 0, 10, 5, 0};

// Create the second rectangle using OpenCASCADE
Rectangle(2) = {5, 5, 0, 10, 10, 0};

// Merge overlapping lines
Line Loop(3) = {4, -5, 6, 7};
Plane Surface(4) = {3};

// Merge overlapping lines for the other rectangle
Line Loop(5) = {1, 2, 3, 5};
Plane Surface(6) = {5};