function poly2gmsh(fname, pv, elemtype, hmin, hmax)
    
    if nargin<3 
        elemtype = 0;
    end

    fid = fopen(fname, 'w');

    loops = local_split_loops(pv);
    nloops = numel(loops);
    npv = sum(cellfun(@(p) size(p,1), loops));
    
    pointId = 0;
    if nargin>3
        for iloop = 1:nloops
            p = loops{iloop};
            for i = 1:size(p,1)
                pointId = pointId + 1;
                fprintf(fid, 'Point(%d) = {%g, %g, 0};\n', pointId, [p(i,:)]);
            end
        end
    else
        for iloop = 1:nloops
            p = loops{iloop};
            for i = 1:size(p,1)
                pointId = pointId + 1;
                fprintf(fid, 'Point(%d) = {%g, %g, 0};\n', pointId, p(i,:));
            end
        end
    end
    fprintf(fid, '\n');

    lineId = 0;
    startPoint = 1;
    loopIds = zeros(1, nloops);
    for iloop = 1:nloops
        n = size(loops{iloop}, 1);
        lines = zeros(1, n);
        for i = 1:n
            lineId = lineId + 1;
            p1 = startPoint + i - 1;
            p2 = startPoint + mod(i, n);
            fprintf(fid, 'Line(%d) = {%d, %d};\n', lineId, p1, p2);
            lines(i) = lineId;
        end
        loopIds(iloop) = npv + iloop;
        fprintf(fid, 'Line Loop(%d) = {%s};\n', loopIds(iloop), local_int_list(lines));
        startPoint = startPoint + n;
    end
    fprintf(fid, '\n');

    fprintf(fid, 'Plane Surface(1) = {%s};\n', local_int_list(loopIds));
    if elemtype==1
        % Transfinite surfaces are only valid for simple 3/4-corner patches.
        % General polygons and surfaces with holes must use unstructured
        % meshing followed by recombination.
        if nloops == 1 && npv == 4
            fprintf(fid, 'Transfinite Surface {1};\n');
        end
        fprintf(fid, 'Recombine Surface {1};\n');
        fprintf(fid, 'Mesh.RecombineAll = 1;\n'); 
        fprintf(fid, 'Mesh.RecombinationAlgorithm = 2;\n'); 
        %fprintf(fid, 'Mesh.SubdivisionAlgorithm = 1;\n');     
    else
     %fprintf(fid, 'Plane Surface(1) = {%d};\n', npv);
     fprintf(fid, 'Recombine Surface {1};\n');    
     %fprintf(fid, 'Mesh.RecombineAll = 1;\n');    
    end

    % fprintf(fid, 'Mesh.MeshSizeMin = %g;\n', hmin); 
    % fprintf(fid, 'Mesh.MeshSizeMax = %g;\n', hmax); 
    fclose(fid);

end

function loops = local_split_loops(pv)
    if size(pv,2) ~= 2
        error('pv must be an n-by-2 array.');
    end
    sep = any(isnan(pv), 2);
    breaks = [0; find(sep); size(pv,1)+1];
    loops = {};
    for i = 1:(numel(breaks)-1)
        p = pv((breaks(i)+1):(breaks(i+1)-1), :);
        if isempty(p)
            continue;
        end
        if any(isnan(p(:)))
            error('NaN values are only allowed as full separator rows.');
        end
        if size(p,1) > 1 && norm(p(1,:) - p(end,:)) <= 100*eps(max(1, norm(p(1,:))))
            p = p(1:end-1,:);
        end
        if size(p,1) < 3
            error('Each polygon loop must contain at least three points.');
        end
        loops{end+1} = p; %#ok<AGROW>
    end
    if isempty(loops)
        error('pv does not contain any polygon loops.');
    end
end

function str = local_int_list(values)
    str = sprintf('%d,', values);
    str = str(1:end-1);
end
