"""Read Sharp-B equilibrium-air wall surfacequantities and optionally write plots."""

include(joinpath(dirname(@__DIR__), "isoq2d_equichem", "postprocess_surfacequantities.jl"))

function try_write_surface_plots(result)
    try
        @eval import Plots
    catch err
        println("Plots.jl not available; skipping PNG surface plots ($err).")
        return
    end

    plotdir = result[:plotdir]
    s = result[:s]
    x = result[:x][:, 1]
    y = result[:x][:, 2]
    fields = [
        (:Cp, "C_p", "Pressure coefficient, C_p"),
        (:Cf, "C_f", "Skin-friction coefficient, C_f"),
        (:Cq, "C_q", "Heat-flux coefficient, C_q"),
    ]

    for (key, ylabel, title) in fields
        values = result[key]
        lineplot = Plots.plot(
            s, values;
            marker=:circle,
            linewidth=1.0,
            xlabel="wall arclength",
            ylabel=ylabel,
            title=title,
            grid=true,
            legend=false,
        )
        Plots.savefig(lineplot, joinpath(plotdir, "$(String(key))_vs_arclength.png"))

        scatterplot = Plots.scatter(
            x, y;
            zcolor=values,
            markerstrokewidth=0,
            aspect_ratio=:equal,
            xlabel="x",
            ylabel="y",
            title="$(ylabel) on saved wall points",
            legend=false,
            colorbar=true,
        )
        Plots.savefig(scatterplot, joinpath(plotdir, "$(String(key))_wall_points.png"))
    end
end

function postprocess_sharpb2_surfacequantities(pde; wallib::Int=3)
    result = postprocess_surfacequantities(pde; wallib=wallib)
    try_write_surface_plots(result)
    return result
end
