% Convolutional Neural Network Segmentation %

function outd=kmean(im)
%a3=im(:,:,2);
%figure,imshow(im);
%title('grayscale image');
a=im2double(im);
[m n]=size(a);
k=4;
cc1=0.49;
cc2=0.30; 
cc3=0.7; 
cc4=0.97;
A=cat(3,a,a,a,a);
iteration=0;
while(iteration<5)
    c1=repmat(cc1,m,n);
    c2=repmat(cc2,m,n);
    c3=repmat(cc3,m,n);
    c4=repmat(cc4,m,n);
    c=cat(3,c1,c2,c3,c4);
    distance=(A-c).^2;
    d1=distance(:,:,1);
    d2=distance(:,:,2);
    d3=distance(:,:,3);
    d4=distance(:,:,4);
    M=cat(3,d1,d2,d3,d4);
    y=zeros(m,n);
    y=im2double(y);
    for i=1:m
        for j= 1:n
            if min(M(i,j,:))==d1(i,j)
                y(i,j)=1;
            else if min(M(i,j,:))==d2(i,j)
                y(i,j)=2;
            else if min(M(i,j,:))==d3(i,j)
                y(i,j)=3;
            else if min(M(i,j,:))==d4(i,j)
                y(i,j)=4;
                end
                end
                end
            end
        end
    end
    out1=zeros(m,n);
    out2=zeros(m,n);
    out3=zeros(m,n);
    out4=zeros(m,n);
    for i=1:m
        for j= 1:n
            if y(i,j)==1
                out1(i,j)=a(i,j);
            else if y(i,j)==2
                out2(i,j)=a(i,j);
            else if y(i,j)==3
                 out3(i,j)=a(i,j);
            else if y(i,j)==4
                out4(i,j)=a(i,j);
                end
                end
                end
            end
        end
 end   
 % Mean value for Iteration 1
 count1=0;
 sum1=0;
 for i=1:m
     for j= 1:n
         if out1(i,j)~=0
             count1=count1+1;
             sum1=sum1+out1(i,j);
         end
     end
 end
 cc1=sum1/count1;
  % Mean value for Iteration 2
 count2=0;
 sum2=0;
 for i=1:m
     for j= 1:n
         if out2(i,j)~=0
             count2=count2+1;
             sum2=sum2+out2(i,j);
                
            end
     end
 end
 cc2=sum2/count2;
  % Mean value for Iteration 3
 count3=0;
 sum3=0;
 for i=1:m
     for j= 1:n
         if out3(i,j)~=0
             count3=count3+1;
             sum3=sum3+out3(i,j);
         end
     end
 end
 cc3=sum3/count3;
  % Mean value for Iteration 4
 count4=0;
 sum4=0;
 for i=1:m
     for j= 1:n
         if out4(i,j)~=0
            count4=count4+1;
            sum4=sum4+out4(i,j);
         end
     end
 end
 cc4=sum4/count4;
 iteration=iteration+1;
end

%%%OPTIC DISK EXTRACTION%%%
for i = 1:m
    for j=1:n
        if out3(i,j)>0
            out5(i,j)=1;
        else
            out5(i,j)=0;
        end
    end
end


for i = 1:m
    for j=1:n
        if out4(i,j)>0.2
            outb(i,j)=1;
        else
            outb(i,j)=0;
        end
    end
end

p1=imfill(outb,'holes');

se = strel('disk',2);
outd=imdilate(out5,se);
outd=bwareaopen(outd,500);
figure,imshow(outd)
title('segmentation');
pause(2);
end
 