tic;
I = imread('T2hreees.bmp');
%I = rgb2gray(I);
I = imadjust(I);
imshow(I);
hy = fspecial('sobel');
hx = hy';
Iy = imfilter(double(I), hy, 'replicate');
Ix = imfilter(double(I), hx, 'replicate');
gradmag = sqrt(Ix.^2 + Iy.^2);
imshow(gradmag,[]);
L = watershed(gradmag);
Lrgb = label2rgb(L);
figure, imshow(Lrgb);
se = strel('disk',5);
Io = imopen(I, se);
figure
imshow(Io);
Ie = imerode(I, se);
Iobr = imreconstruct(Ie, I);
figure
imshow(Iobr);
Ioc = imclose(Io, se);
figure
imshow(Ioc);
Iobrd = imdilate(Iobr, se);
Iobrcbr = imreconstruct(imcomplement(Iobrd), imcomplement(Iobr));
Iobrcbr = imcomplement(Iobrcbr);
figure
imshow(Iobrcbr);
fgm = imregionalmax(Iobrcbr);
figure
imshow(fgm);
I2 = I;
I2(fgm) = 255;
figure
imshow(I2);
se2 = strel(ones(5,5));
fgm2 = imclose(fgm, se2);
fgm3 = imerode(fgm2, se2);
fgm4 = bwareaopen(fgm3, 5);
I3 = I;
I3(fgm4) = 255;
imshow(I3);
bw = imbinarize(Iobrcbr);
figure
imshow(bw);
D = bwdist(bw);
DL = watershed(D);
bgm = DL == 0;
figure
imshow(bgm);
gradmag2 = imimposemin(gradmag, bgm | fgm4);
L = watershed(gradmag2);
I4 = I;
I4(imdilate(L == 0, ones(2, 2)) | bgm | fgm4) = 255;
figure
imshow(I4);
Lrgb = label2rgb(L, 'jet', 'w', 'shuffle');
figure
imshow(Lrgb);
toc;