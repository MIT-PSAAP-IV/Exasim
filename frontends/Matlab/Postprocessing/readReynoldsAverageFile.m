function reavg = readReynoldsAverageFile(filename, npe, nc, neExpected)
fid = fopen(filename, 'r');
if fid < 0
    error('Cannot open Reynolds-average file: %s', filename);
end
cleanup = onCleanup(@() fclose(fid));

data = fread(fid, 'double');
if isempty(data)
    error('Reynolds-average file is empty: %s', filename);
end

if numel(data) >= 3 && isHeader(data(1:3), npe, nc)
    ne = data(3);
    payload = data(4:end);
    if numel(payload) ~= npe * nc * ne
        error('Header payload size mismatch in %s.', filename);
    end
else
    payload = data;
    if mod(numel(payload), npe * nc) ~= 0
        error('Payload size is incompatible with npe=%d and nc=%d in %s.', ...
              npe, nc, filename);
    end
    ne = numel(payload) / (npe * nc);
end

if ne ~= neExpected
    error('Element count mismatch in %s: file has %d, mesh has %d.', ...
          filename, ne, neExpected);
end

reavg = reshape(payload, [npe, nc, ne]);
end

function tf = isHeader(header, npe, nc)
tf = numel(header) == 3 && ...
     all(isfinite(header)) && ...
     all(header == floor(header)) && ...
     header(1) == npe && header(2) == nc && header(3) > 0;
end
