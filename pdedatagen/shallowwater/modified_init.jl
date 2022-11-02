import SpeedyWeather

using SpeedyWeather: ShallowWaterModel, initialize_from_rest, spectral!, gradient_latitude, gradient_longitude

using Parameters: @unpack

"""Initialize prognostic variables from rest or restart from file."""
function SpeedyWeather.initial_conditions(M::ShallowWaterModel)

    @unpack initial_conditions = M.parameters

    if initial_conditions == :rest
        progn = initialize_from_rest(M)

    elseif initial_conditions == :barotropic_vorticity
        progn = initialize_from_rest(M)

        P = M.parameters    # unpack and rename
        G = M.geospectral

        @unpack nlon, nlat, nlev = G.geometry
        @unpack latd, lon, coslat, sinlat, radius_earth = G.geometry
        @unpack lmax, mmax = G.spectral_transform

        # zonal wind
        u_grid1 = @. (25*coslat - 30*coslat^3 + 300*sinlat^2*coslat^6)/coslat+100
        u_grid = repeat(u_grid1',nlon,1)
        u = zeros(Complex{P.NF},lmax+1,mmax+1)
        u = spectral!(u,u_grid,G.spectral_transform)
        ζ = gradient_latitude(u,G.spectral_transform,one_more_l=false,flipsign=true)
        progn.vor[:,:,1,1] .= ζ/radius_earth

        # zonal wave perturbation
        A = 1e-4
        m = 6
        θ0 = 45
        θw = 10

        ζp = convert.(P.NF,A/2*cos.(m*lon)) * convert.(P.NF,coslat .* exp.(-((latd .- θ0)/θw).^2))'
        progn.vor[:,:,1,1] .+= spectral(ζp,G.spectral_transform)

        for k in 2:nlev
            progn.vor[:,:,1,k] .= progn.vor[:,:,1,1]
        end

        # make it less symmetric
        progn.vor[15,1:14,1,:] .+= 5e-6*randn(Complex{P.NF},14,nlev)

    elseif initial_conditions == :barotropic_divergence
        progn = initialize_from_rest(M)

        P = M.parameters    # unpack and rename
        G = M.geospectral

        @unpack nlon, nlat, nlev = G.geometry
        @unpack latd, lon, coslat, sinlat, radius_earth = G.geometry
        @unpack lmax, mmax = G.spectral_transform

        progn.div[5,4,1,1] = 2e-5
        progn.vor[15,1:14,1,:] .+= 3e-6*randn(ComplexF64,14,nlev)


    elseif initial_conditions == :random2
        progn = initialize_from_rest(M)

        P = M.parameters    # unpack and rename
        G = M.geospectral

        @unpack nlon, nlat, nlev = G.geometry
        @unpack latd, lon, coslat, sinlat, radius_earth = G.geometry
        @unpack lmax, mmax = G.spectral_transform

        offset = rand(80:120)
        coeffs = (rand(-20:30), rand(-20:40), rand(-20:40))

        # zonal wind
        #u_grid1 = @. (coeffs[1]*coslat - coeffs[2]*coslat^3 + coeffs[3]*sinlat^2*coslat^6)/coslat+offset
        #u_grid1 = coeffs[1] * rand(P.NF, length(coslat)) .* coslat  .+ offset
        u_grid = vcat([coeffs[1] * rand(P.NF, length(coslat)) .* coslat .- coeffs[2] * coslat.^2 .+ coeffs[3] .* sinlat .* coslat .+ offset for i in 1:nlon]...)
        u_grid = reshape(u_grid, nlon, nlat)
        #u_grid = repeat(u_grid1',nlon,1)
        u = zeros(Complex{P.NF},lmax+1,mmax+1)
        u = spectral!(u,u_grid,G.spectral_transform)
        ζ = gradient_latitude(u,G.spectral_transform,one_more_l=false,flipsign=true)
        progn.vor[:,:,1,1] .= ζ/radius_earth

        # zonal wave perturbation
        A = 1e-4
        m = 6
        θ0 = 45
        θw = 10

        ζp = convert.(P.NF,A/2*cos.(m*lon)) * convert.(P.NF,coslat .* exp.(-((latd .- θ0)/θw).^2))'
        progn.vor[:,:,1,1] .+= spectral(ζp,G.spectral_transform)

        for k in 2:nlev
            progn.vor[:,:,1,k] .= progn.vor[:,:,1,1]
        end

        # make it less symmetric
        progn.vor[15,1:14,1,:] .+= 5e-6*randn(Complex{P.NF},14,nlev)

    elseif initial_conditions == :random3
        progn = initialize_from_rest(M)

        P = M.parameters    # unpack and rename
        G = M.geospectral

        @unpack nlon, nlat, nlev = G.geometry
        @unpack latd, lon, coslat, sinlat, radius_earth = G.geometry
        @unpack lmax, mmax = G.spectral_transform

        offset = rand(80:120)
        coeffs = (rand(-20:30), rand(-20:40), rand(-20:40))
        powers = (rand(2:4), rand(1:4))
        # zonal wind
        #u_grid1 = @. (coeffs[1]*coslat - coeffs[2]*coslat^3 + coeffs[3]*sinlat^2*coslat^6)/coslat+offset
        #u_grid1 = coeffs[1] * rand(P.NF, length(coslat)) .* coslat  .+ offset
        u_grid = vcat([coeffs[1] * rand(P.NF, length(coslat)) .* coslat .- coeffs[2] * coslat .^ powers[1] .+ coeffs[3] .* sinlat .^ powers[2] .* coslat .+ offset for i in 1:nlon]...)
        u_grid = reshape(u_grid, nlon, nlat)
        #u_grid = repeat(u_grid1',nlon,1)
        u = zeros(Complex{P.NF},lmax+1,mmax+1)
        u = spectral!(u,u_grid,G.spectral_transform)
        ζ = gradient_latitude(u,G.spectral_transform,one_more_l=false,flipsign=true)
        progn.vor[:,:,1,1] .= ζ/radius_earth

        # zonal wave perturbation
        A = 1e-4
        m = 6
        θ0 = 45
        θw = 10

        ζp = convert.(P.NF,A/2*cos.(m*lon)) * convert.(P.NF,coslat .* exp.(-((latd .- θ0)/θw).^2))'
        progn.vor[:,:,1,1] .+= spectral(ζp,G.spectral_transform)

        for k in 2:nlev
            progn.vor[:,:,1,k] .= progn.vor[:,:,1,1]
        end

        # make it less symmetric
        progn.vor[15,1:14,1,:] .+= 5e-6*randn(Complex{P.NF},14,nlev)
    elseif initial_conditions == :restart
        progn = initialize_from_file(M)         # TODO this is not implemented yet
    else
        throw(error("Incorrect initialization option, $initial_conditions given."))
    end

    # SCALING
    progn.vor .*= M.geospectral.geometry.radius_earth
    progn.div .*= M.geospectral.geometry.radius_earth

    return progn
end
