function make_plate_mesh(nx=120, ny=48, growth=1.2)
    x = collect(range(-0.5, 0.5, length=nx + 1))
    layers = growth .^ collect(0:ny)
    y = 0.2 .* (layers .- 1.0) ./ (layers[end] - 1.0)
    points = hcat(([xx, yy] for yy in y for xx in x)...)
    elements = Matrix{Int}(undef, 4, nx * ny)
    k = 1
    for j in 0:ny-1, i in 0:nx-1
        lower_left = 1 + i + j * (nx + 1)
        elements[:, k] = [lower_left, lower_left + 1, lower_left + nx + 2, lower_left + nx + 1]
        k += 1
    end
    points, elements
end
