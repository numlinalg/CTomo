using LinearAlgebra, SparseArrays, TestImages, Images, Plots

function kaczmarz(A, b, k)
    m, n = size(A)  # Get the size of A
    x = zeros(n)    # Initial guess x^(0)
    for iteration in 1:k
        i = (iteration - 1) % m + 1  # Cycle through rows of A
        a_i = A[i, :]                # i-th row of A
        b_i = b[i]                   # i-th component of b
        x += (b_i - dot(a_i, x)) / dot(a_i, a_i) * a_i  # Kaczmarz update
    end
    return x
end

function parallel_tomo(N, theta, p)
    d = p - 1
    A = get_or_apply_system_matrix(N, theta, p, d)
    x = Float32.(TestImages.shepp_logan(256))
    x = imresize(x, (N, N))
    b = A * reshape(x, N^2)
    return A, b, x
end

function get_or_apply_system_matrix(N, theta, p, d)
    # Define the number of angles
    nA = length(theta)

    # The starting values both the x and the y coordinates
    x0 = LinRange(-d/2, d/2, p)
    y0 = zeros(p)

    # The intersection lines
    x = -N/2:N/2
    y = x

    # Initialize vectors that contain the row numbers, the column numbers,
    # and the values for creating the matrix A efficiently
    rows = zeros(Int, 2*N*nA*p)
    cols = similar(rows)
    vals = zeros(Float64, length(rows))
    idxend = 0

    for i in 1:nA
        x0theta = cosd(theta[i]) .* x0 .- sind(theta[i]) .* y0
        y0theta = sind(theta[i]) .* x0 .+ cosd(theta[i]) .* y0

        a = -sind(theta[i])
        b = cosd(theta[i])

        for j in 1:p
            tx = (x .- x0theta[j]) ./ a
            yx = b .* tx .+ y0theta[j]

            ty = (y .- y0theta[j]) ./ b
            xy = a .* ty .+ x0theta[j]

            t = [tx; ty]
            xxy = [x; xy]
            yxy = [yx; y]

            # Sort the coordinates according to intersection time
            sorted_indices = sortperm(t)
            xxy = xxy[sorted_indices]
            yxy = yxy[sorted_indices]

            # Skip the points outside the box
            valid_indices = (xxy .>= -N/2) .& (xxy .<= N/2) .& (yxy .>= -N/2) .& (yxy .<= N/2)
            xxy = xxy[valid_indices]
            yxy = yxy[valid_indices]

            # Skip double points
            diffs_x = abs.(diff(xxy)) .<= 1e-10
            diffs_y = abs.(diff(yxy)) .<= 1e-10
            diffs = diffs_x .& diffs_y
            if any(diffs)
                xxy = xxy[[true; .!diffs]]
                yxy = yxy[[true; .!diffs]]
            end

            aval = sqrt.(diff(xxy).^2 + diff(yxy).^2)
            col = Int[]

            if !isempty(aval)
                if !((b == 0 && abs(y0theta[j] - N/2) < 1e-15) ||
                     (a == 0 && abs(x0theta[j] - N/2) < 1e-15))
                    xm = 0.5 .* (xxy[1:end-1] .+ xxy[2:end]) .+ N/2
                    ym = 0.5 .* (yxy[1:end-1] .+ yxy[2:end]) .+ N/2

                    col = floor.(Int, xm) .* N .+ (N .- floor.(Int, ym))
                end
            end

            if !isempty(col)
                idxstart = idxend + 1
                idxend = idxstart + length(col) - 1
                idx = idxstart:idxend

                rows[idx] .= (i-1)*p + j
                cols[idx] .= col
                vals[idx] .= aval
            end
        end
    end

    # Truncate excess zeros
    rows = rows[1:idxend]
    cols = cols[1:idxend]
    vals = vals[1:idxend]

    # Create sparse matrix A from the stored values
    A = sparse(rows, cols, vals, p*nA, N^2)
    return A
end

# Set the parameters for the test problem
N = 50            # The image size will be N x N
theta = 0:2:178   # Angles used for projections
p = 75            # Number of parallel rays

A, b, x = parallel_tomo(N, theta, p)

# Solve the test problem
k = 5000
x_s = kaczmarz(A, b, k)

# reshape the result to an image
xs_image = reshape(x_s, N, N)

# Create subplots for "before" and "after" images
p1 = heatmap(x, color=:grays, aspect_ratio=1, title="Original")
p2 = heatmap(xs_image, color=:grays, aspect_ratio=1, title="Kaczmarz")

# Combine the plots into one figure
p = plot(p1, p2, layout=(1, 2), size=(800, 400))

# Display the figure
display(p)
