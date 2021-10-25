% Used for Experiments. See Report.
function [features] = get_patch_features(image, x, y, descriptor_window_image_width)
   n = size(x, 1);
   features = zeros(n, 256);
   
   padding = descriptor_window_image_width/2;
   image = padarray(image, [padding padding], 'symmetric');
   x = x+padding;
   y = y+padding;
   
   for i=1:n
       patch = image(y(i)-padding:y(i)+padding-1, x(i)-padding:x(i)+padding-1, :);
       feature = reshape(patch, [1 256]);
       feature = feature./norm(feature);
       features(i,:) = feature(1,:);
   end
end
