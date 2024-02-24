# Adaptation of
# https://github.com/hanyoseob/matlab-ART
using RadonKA, Images, ImageTransformations, Random, TestImages, ImageIO, ImageView, Distributions

function ART(A, AT, b, x, lambda, niter, bpos)
    ATA = AT(A(ones(Float32, size(x, 1), size(x, 2))))
    for i = 1:niter
        println("Iteration: ", i)
        x .= x .+ lambda .* AT(b .- A(x)) ./ ATA
        if bpos
            x[x .< 0] .= 0
        end
    end
    return x
end

# Julia equivalent of system setting
N = 512
N = 256
ANG = 180
VIEW = 360
THETA = range(0, stop=ANG, length=VIEW)[1:end-1] * (π / 180) # Convert to radians

# Functions to simulate A and AT using RadonKA
A = x -> radon(x, THETA)
AT = y -> backproject(y, THETA) * (π / (2 * length(THETA)))
AINV = y -> backproject(y, THETA)

# Example image loading and resizing (replace with actual image loading)
# img = load("path_to_your_image.png")
# x = imresize(img, (N, N))
# For demonstration, using a test image
x = Float32.(TestImages.shepp_logan(256))
x = imresize(x, (N, N))

# Generating projections
p = A(x)

# Simulated low-dose sinogram generation
i0 = 5e4
pn = exp.(-p)
pn = i0 .* pn
pn = rand.(Poisson.(pn))
pn = max.(-log.(max.(pn, 1) / i0), 0)

# Noise
# NOTE: I am doing something wrong with the noise
#y = pn

# Noiseless
y = p

# ART initialization
x_low = AINV(y) # Initial approximation using backprojection
x0 = zeros(size(x))
lambda = 1.0
niter = 20
bpos = true


# Running ART
x_art = ART(A, AT, y, x0, lambda, niter, bpos)

# plotting
imshow(x_art)

# display original image
# imshow(x)

# display sinogram
# imshow(y)

# display initial approximation
# imshow(x_low)
