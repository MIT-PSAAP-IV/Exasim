function result = phase10_text2code(k, orientation)
%PHASE10_TEXT2CODE Mixed triangle/quad MATLAB-to-Text2Code round trip.
if nargin < 1, k = 1; end
if nargin < 2, orientation = 'tri-quad'; end
result = phase9_text2code(k, 'mixedmeshes', orientation);
fprintf('PASS phase10_text2code %s %d+%d\n', orientation, k, k);
end
