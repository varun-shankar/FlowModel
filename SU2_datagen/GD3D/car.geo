cx = 4; cz = 0;
//+
SetFactory("OpenCASCADE");
//+
Box(1) = {0, 0, -2.5, 20, 5, 5};
//+
Point(17) = {1, .75, 1, 1.0};
//+
Point(18) = {2, .8, 1, 1.0};
//+
Point(19) = {3, 1.5, 1, 1.0};
//+
Point(20) = {5, 1.5, 1, 1.0};
//+
Point(21) = {6, 1, 1, 1.0};
//+
Point(22) = {6, 0, 1, 1.0};
//+
Point(23) = {1, 0, 1, 1.0};
//+
Line(13) = {23, 17};
//+
Line(14) = {17, 18};
//+
Line(15) = {18, 19};
//+
Line(16) = {19, 20};
//+
Line(17) = {20, 21};
//+
Line(18) = {21, 22};
//+
Line(19) = {22, 23};
//+
Curve Loop(7) = {13, 14, 15, 16, 17, 18, 19};
//+
Plane Surface(7) = {7};
//+
Extrude {0, 0, -2} {
  Surface{7}; 
}
//+
Fillet{2}{21, 29}{0.5}
//+
Fillet{2}{37, 39}{0.2}
//+
Fillet{2}{25}{0.8}
//+
Fillet{2}{35, 30, 27, 28, 33, 36, 40, 43, 38, 46, 48, 21, 23, 24, 17, 15, 13, 14, 16, 18, 20}{0.2}
//+
BooleanDifference{ Volume{1}; Delete; }{ Volume{2}; Delete; }
//+
Fillet{1}{29, 23}{0.2}
//+
Field[1] = Box;
//+
Field[1].VIn = 0.08;
//+
Field[1].VOut = 0.7;
//+
Field[1].XMax = 10;
//+
Field[1].YMax = 2;
//+
Field[1].YMin = 0;
//+
Field[1].ZMax = 2;
//+
Field[1].ZMin = -2;
//+
Field[1].Thickness = 5;
//+
Background Field = 1;


//+
Physical Surface("ground") = {1};
//+
Physical Surface("inlet") = {12};
//+
Physical Surface("outlet") = {11};
//+
Physical Surface("topsides") = {13, 22, 10};
//+
Physical Surface("car") = {45, 48, 42, 38, 21, 9, 7, 19, 27, 29, 5, 31, 41, 43, 30, 28, 24, 26, 3, 23, 39, 40, 44, 49, 47, 36, 34, 32, 25, 4, 6, 18, 37, 14, 16, 2, 15, 35, 46, 33, 20, 8, 17};

//+
Sphere(2) = {cx, 1.5, cz, 0.1, -Pi/2, Pi/2, 2*Pi};
//+
Dilate {{cx, 1.5, cz}, {1, 2, 1}} {
  Volume{2}; 
}
//+
BooleanDifference{ Volume{1}; Delete; }{ Volume{2}; Delete; }
//+
Physical Volume("internal") = {1};
//+
Physical Surface("lidar") = {50};