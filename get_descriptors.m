% Local Feature Stencil Code
% Written by James Hays for CS 143 @ Brown / CS 4476/6476 @ Georgia Tech

% Returns a set of feature descriptors for a given set of interest points. 

% 'image' can be grayscale or color, your choice.
% 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
%   The local features should be centered at x and y.
% 'descriptor_window_image_width', in pixels, is the local feature descriptor width. 
%   You can assume that descriptor_window_image_width will be a multiple of 4 
%   (i.e., every cell of your local SIFT-like feature will have an integer width and height).
% If you want to detect and describe features at multiple scales or
% particular orientations, then you can add input arguments.

% 'features' is the array of computed features. It should have the
%   following size: [length(x) x feature dimensionality] (e.g. 128 for
%   standard SIFT)

function [features] = get_features(image, x, y, descriptor_window_image_width)

% To start with, you might want to simply use normalized patches as your
% local feature. This is very simple to code and works OK. However, to get
% full credit you will need to implement the more effective SIFT descriptor
% (See Szeliski 4.1.2 or the original publications at
% http://www.cs.ubc.ca/~lowe/keypoints/)

% Your implementation does not need to exactly match the SIFT reference.
% Here are the key properties your (baseline) descriptor should have:
%  (1) a 4x4 grid of cells, each descriptor_window_image_width/4. 'cell' in this context
%    nothing to do with the Matlab data structue of cell(). It is simply
%    the terminology used in the feature literature to describe the spatial
%    bins where gradient distributions will be described.
%  (2) each cell should have a histogram of the local distribution of
%    gradients in 8 orientations. Appending these histograms together will
%    give you 4x4 x 8 = 128 dimensions.
%  (3) Each feature should be normalized to unit length
%
% You do not need to perform the interpolation in which each gradient
% measurement contributes to multiple orientation bins in multiple cells
% As described in Szeliski, a single gradient measurement creates a
% weighted contribution to the 4 nearest cells and the 2 nearest
% orientation bins within each cell, for 8 total contributions. This type
% of interpolation probably will help, though.

% You do not have to explicitly compute the gradient orientation at each
% pixel (although you are free to do so). You can instead filter with
% oriented filters (e.g. a filter that responds to edges with a specific
% orientation). All of your SIFT-like feature can be constructed entirely
% from filtering fairly quickly in this way.

% You do not need to do the normalize -> threshold -> normalize again
% operation as detailed in Szeliski and the SIFT paper. It can help, though.

% Another simple trick which can help is to raise each element of the final
% feature vector to some power that is less than one.

    n = size(x, 1);
    features = zeros(n, (descriptor_window_image_width/4).^2.*8);

    % Sobel filters for all 8 edge directions: f(degrees)
    f0 = [1 2 1; 0 0 0; -1 -2 -1];
    f45 = [0 1 2; -1 0 1; -2 -1 0];
    f90 = [-1 0 1; -2 0 2; -1 0 1];
    f135 = [-2 -1 0; -1 0 1; 0 1 2];
    f180 = [-1 -2 -1; 0 0 0; 1 2 1];
    f225 = [0 -1 -2; 1 0 -1; 2 1 0];
    f270 = [1 0 -1; 2 0 -2; 1 0 -1];
    f315 = [2 1 0; 1 0 -1; 0 -1 -2];

    % Padding to account for the 16x16 window
    padding = descriptor_window_image_width/2;
    image = padarray(image, [padding padding], 'symmetric');
    x = x+padding;
    y = y+padding;

    f = cat(3, f0, f45, f90, f135, f180, f225, f270, f315);

    % Create a (padded_image_height x padded_image_width x 8) edge response
    % matrix
    [s1, s2, ~] = size(image);
    edges = zeros(s1,s2,8);
    for i=1:8
        edges(:,:,i) = imfilter(image, f(:,:,i));
    end

    % Extract Features
    for i=1:n
        patch = edges(y(i)-padding:y(i)+padding-1, x(i)-padding:x(i)+padding-1, :);
        gaussian = fspecial('gaussian', [descriptor_window_image_width descriptor_window_image_width], descriptor_window_image_width/2);
        gauss_patch = patch.*gaussian;
        bin_width = descriptor_window_image_width/4;
        sum_bin = zeros(4,4,8);
        for j=1:4
            for k=1:4
                start_x = (j-1)*bin_width + 1;
                end_x = j*bin_width;
                start_y = (k-1)*bin_width + 1;
                end_y = k*bin_width;
                bin = gauss_patch(start_x:end_x,start_y:end_y,:);
                sum_bin(j,k,:) = sum(bin, [1 2]);
            end
        end
        feature = reshape(sum_bin, [1 128]);
        feature = feature./norm(feature);
        features(i,:) = feature(1,:);
    end
end




