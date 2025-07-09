I = imread('108.png'); 
bw = imbinarize(I, graythresh(I));
bw2 = imfill(bw,'holes');
imshow(bw2);
figure(2);
L = bwlabel(bw2,4);
imshow(label2rgb(L, spring, [.2 .2 .2]))
s = regionprops(L,'all');
