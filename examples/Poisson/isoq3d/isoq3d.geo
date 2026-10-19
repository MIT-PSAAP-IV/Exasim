SetFactory("OpenCASCADE");
Mesh.CharacteristicLengthMin = 0.025;
Mesh.CharacteristicLengthMax = 0.06;
Cylinder(1) = {0, 0, 0, 1, 0, 0, 0.17};
Cylinder(2) = {0, 0, 0, 1, 0, 0, 0.05};
shell[] = BooleanDifference{ Volume{1}; Delete; }{ Volume{2}; Delete; };
Box(3) = {0, 0, 0, 1, 0.17, 0.17};
quarter[] = BooleanIntersection{ Volume{shell[]}; Delete; }{ Volume{3}; Delete; };
