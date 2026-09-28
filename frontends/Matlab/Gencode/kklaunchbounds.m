function n = kklaunchbounds(kkdir)
% Emit the generated Kokkos kernels with HIP LaunchBounds<64,1>.
%
% On MI300A (CDNA3) the default ~256-thread workgroups launch too few workgroups for the per-block
% kernel sizes to fill the 228 CUs; 1-wavefront workgroups give ~4x more of them. The generated
% kernels are not register-limited there, so the smaller workgroups cost nothing. Guarded by
% KOKKOS_ENABLE_HIP: CUDA and CPU builds keep the default policy. Same rewrite as the Python
% generator (gencodeelem.py and friends), applied here to every generated launch of the form
%   Kokkos::parallel_for("Name", ng, KOKKOS_LAMBDA(const size_t i) {
% Returns the number of launches rewritten.

tab = char(9); nl = newline;
rep = ['Kokkos::parallel_for("$1",' nl ...
       '#if defined(KOKKOS_ENABLE_HIP)' nl ...
       tab tab 'Kokkos::RangePolicy<Kokkos::LaunchBounds<64,1>>(0, (size_t)ng),' nl ...
       '#else' nl ...
       tab tab 'ng,' nl ...
       '#endif' nl ...
       tab tab 'KOKKOS_LAMBDA(const size_t i) {'];
pat = 'Kokkos::parallel_for\("(\w+)", ng, KOKKOS_LAMBDA\(const size_t i\) \{';

n = 0;
files = dir(fullfile(char(kkdir), '*.cpp'));
for i = 1:numel(files)
    fn = fullfile(files(i).folder, files(i).name);
    s = fileread(fn);
    k = numel(regexp(s, pat));
    if k > 0
        s = regexprep(s, pat, rep);
        fid = fopen(fn, 'w');
        fwrite(fid, s);
        fclose(fid);
        n = n + k;
    end
end
end
