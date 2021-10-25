% Local Feature Stencil Code
% Written by James Hays for CS 143 @ Brown / CS 4476/6476 @ Georgia Tech

% Please implement the "nearest neighbor distance ratio test", 
% Equation 4.18 in Section 4.1.3 of Szeliski. 
% For extra credit you can implement spatial verification of matches.

%
% Please assign a confidence, else the evaluation function will not work.
%

% This function does not need to be symmetric (e.g., it can produce
% different numbers of matches depending on the order of the arguments).

% Input:
% 'features1' and 'features2' are the n x feature dimensionality matrices.
%
% Output:
% 'matches' is a k x 2 matrix, where k is the number of matches. The first
%   column is an index in features1, the second column is an index in features2. 
%
% 'confidences' is a k x 1 matrix with a real valued confidence for every match.

function [matches, confidences] = match_features(features1, features2)

% Placeholder random matches and confidences.
num_features = size(features1, 1);
matches = zeros(num_features, 2);
confidences = zeros(num_features, 1);

for i=1:num_features
    tiled_feature = repmat(features1(i,:), size(features2, 1), 1);
    diff_matrix = tiled_feature - features2;
    % Euclidean Distance
    dist_matrix = sqrt(diag(diff_matrix * diff_matrix.'));
    [dist_matrix, idx] = sort(dist_matrix);
    % Confidence Calculation
    nearest_two = dist_matrix(1:2);
    matches(i, :) = [i, idx(1)];
    confidences(i, :) = 1-nearest_two(1)/nearest_two(2);
end

% Uncomment below to only output 100 most confident matches
% [confidences, idx] = sort(confidences, 'descend');
% matches = matches(idx, :);
% confidences = confidences(1:100);
% matches = matches(1:100, :);

% Remember that the NNDR test will return a number close to 1 for 
% feature points with similar distances.
% Think about how confidence relates to NNDR.