function lower = lowerSurfaceMask(x,y,proftype, xyMidChord)
switch proftype
    case 0
        lower = y < 0;
    case 1
        lower = y < getMeanLine(x, xyMidChord);
    otherwise
        error('Unsupported proftype = %d.', proftype);
end
end

function yMeanLine = getMeanLine(x, xyMidChord)
yMeanLine = interp1(xyMidChord(:,1),xyMidChord(:,2),x,'linear','extrap');
end
