clc;clear;

image_size = 32;
compress_size = 16;
radio = (compress_size)/(image_size*image_size);
img_num = 6;
%matrix = load('./matrix/matrix.mat','fcw_1');
%measure_max = cell2mat(struct2cell(matrix));
%measure_max = measure_max';
measure_max =randn(compress_size,image_size*image_size);
measure_max = (measure_max>0);
switch img_num
    case 1
        im_org = rgb2gray(imread('./data/baby.bmp')); 
    case 2
        im_org = imread('./data/bridge.bmp'); 
    case 3
        im_org = rgb2gray(imread('./data/lenna.bmp')); 
    case 4
        im_org = rgb2gray(imread('./data/man.bmp')); 
    case 5
        im_org = rgb2gray(imread('./data/pepper.bmp')); 
    case 6
        im_org = imread('./data/head.bmp'); 
end
        
clear opts
opts.mu = 2^7;
opts.beta = 2^5;
opts.tol = 1E-3;
opts.maxit = 300;
opts.TVnorm = 1;
opts.nonneg = true;

im_size =  size(im_org);
h = im_size(1);w = im_size(2);

nx = 0;ny = 0;
for a=1:image_size:h-image_size+2
    nx = nx + 1;ny = 0;
    for b=1:image_size:w-image_size+2
        ny = ny + 1;
    end
end
img_reb = uint8(zeros(nx*image_size,ny*image_size, 1));
im_org_ = img_reb;
%starting.......................................................
for x=0:image_size:h-image_size+2
    x = x + 1; y = 0;
    for y=0:image_size:w-image_size+2
        y = y + 1;
        img_div = im_org(x:x+image_size-1, y:y+image_size-1,1);
        im_org_(x:x+image_size-1, y:y+image_size-1,1) = img_div;
        img_div = reshape(img_div,image_size*image_size,1);
        img_div = double(img_div(:,1));
        measure_data = measure_max*img_div;
        OUT = TVAL3(measure_max,measure_data,image_size,image_size,opts);
        img_reb(x:x+image_size-1, y:y+image_size-1,1) = OUT;
    end
end

imwrite(im_org_,'./result/ORG.bmp');imwrite(img_reb,'./result/TVAL3.bmp');

subplot(121); imshow(im_org_,[0,255]);
title('Original','fontsize',18); drawnow;
xlabel(sprintf('Radio:%f',radio),'fontsize',16);

subplot(122); imshow(img_reb,[0,255]);
title('Recovered by TVAL3','fontsize',18);
final_psnr = psnr(im_org_,img_reb);
final_ssim = ssim(im_org_,img_reb);
xlabel({sprintf('PSNR:%f',final_psnr);sprintf('SSIM:%f',final_ssim)},'fontsize',12);