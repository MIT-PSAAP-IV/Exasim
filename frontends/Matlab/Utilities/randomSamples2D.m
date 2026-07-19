function p = randomSamples2D(N, xmin, xmax, ymin, ymax)
%RANDOMSAMPLES2D Generate Latin hypercube samples in a rectangle.
%
%   p = randomSamples2D(N, xmin, xmax, ymin, ymax)
%
% Inputs:
%   N     - Number of sample points
%   xmin  - Minimum x-coordinate
%   xmax  - Maximum x-coordinate
%   ymin  - Minimum y-coordinate
%   ymax  - Maximum y-coordinate
%
% Output:
%   p     - N-by-2 array of sample points. Each row is [x, y].
%
% Example:
%   p = randomSamples2D(1000, -1, 2, -0.5, 1.5);
%
%   figure;
%   plot(p(:,1), p(:,2), '.');
%   axis equal;
%   xlim([xmin xmax]);
%   ylim([ymin ymax]);
%   grid on;

    % Input validation
    validateattributes(N, {'numeric'}, ...
        {'scalar','integer','positive'}, mfilename, 'N', 1);

    validateattributes(xmin, {'numeric'}, ...
        {'scalar','real'}, mfilename, 'xmin', 2);

    validateattributes(xmax, {'numeric'}, ...
        {'scalar','real','>',xmin}, mfilename, 'xmax', 3);

    validateattributes(ymin, {'numeric'}, ...
        {'scalar','real'}, mfilename, 'ymin', 4);

    validateattributes(ymax, {'numeric'}, ...
        {'scalar','real','>',ymin}, mfilename, 'ymax', 5);

    % Latin hypercube samples in the unit square
    u = lhsdesign(N, 2);

    % Map to the physical domain
    p = zeros(N,2);
    p(:,1) = xmin + (xmax - xmin) * u(:,1);
    p(:,2) = ymin + (ymax - ymin) * u(:,2);

end