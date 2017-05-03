%cut = imread('C1-cuttlefish750-pupilThresholdedSmoothed.tif'); 
%cutEro1 = imread('C1-cuttlefish750-pupilThresholdedEroded1Smth.tif');
% 

%cut = imread('C1-cuttlefish750-pupilThresholded2.tif'); 
%cutEro1 = imread('C1-cuttlefish750-pupilThresholdedEroded1.tif'); 
%cutEro2 = imread('C1-cuttlefish750-pupilThresholdedEroded2.tif'); 
%cutDil1 = imread('C1-cuttlefish750-pupilThresholdedDilated1.tif');
%    cutFT = fftshift(fft2(cut));
%     cutEro1FT = fftshift((fft2(cutEro1)));
%     cutEro2FT = fftshift((fft2(cutEro2)));
%     cutDil1FT = fftshift((fft2(cutDil1)));
function [imgStack,imgStackRGB,PSF] = cuttlefish(cSpher,fN,nStk,off,defAtt)
    
    img = imread(fN);
    sBNarrower = imread('sBandiensisNarrower-1.tif');
    sBNarrow = imread('sBandiensisNarrow-1.tif');
    sBMid = imread('sBandiensisMid-1.tif');
    sBFull = imread('sBandiensisFull-1.tif');
    sBFuller = imread('sBandiensisFuller-1.tif');
    
    dimX = size(sBNarrow,1);
    mid = dimX/2;
    
    %set up aperture radius
    N = 12.5; %25 mm eye 
    [X1,Y1] = meshgrid(-N:2*N/dimX:N-N/dimX,-N:2*N/dimX:N-N/dimX);
    %figure(); imstretch(X1);
    %figure(); imstretch(Y1);
    rad = sqrt((X1.^2 + Y1.^2));
    %figure(); imstretch(rad);
    %imshow(rad<.5);
    %add defocus
    apDef = exp(sqrt(3) * (2*rad.^2 - 1));
    apSpher = exp(sqrt(5) * (6*rad.^4 - 6* rad.^2 + 1));
    %cSpher = 0.1; 
    %add pupil mask, defocus (wavelength dependent) and spherical
    %aberrationfN =
    %imgStack = zeros(dimX+off*2,dimX+off*2,nStk+1);
    imgStack = zeros(dimX,dimX,nStk+1);
    PSF = zeros(dimX,dimX,3,nStk+1);
    %PSF = zeros(off*2+1,off*2+1,3,9);
    %nStk = 10;
    for ck = 1:nStk
        cChromR = (ck-1) / (nStk-1);
        cChromB = 1.0 - cChromR;
        cChromG = 0.5 - cChromR;
        %size(exp(cChromR*apDef+cSpher*apSpher))
        %size(sBNarrower)
       
        apR1 = double(sBNarrow) .* exp(j*2*pi*(cChromR/defAtt*apDef+cSpher*apSpher));
        apG1 = double(sBNarrow) .* exp(j*2*pi*(cChromG/defAtt*apDef+cSpher*apSpher));
        apB1 = double(sBNarrow) .* exp(j*2*pi*(cChromB/defAtt*apDef+cSpher*apSpher));
        
        %apR1 = double(sBNarrow) .* exp(j*2*pi*cChromR/5000*apDef+cSpher*apSpher);
        %apG1 = double(sBNarrow) .* exp(j*2*pi*cChromG/5000*apDef+cSpher*apSpher);
        %apB1 = double(sBNarrow) .* exp(j*2*pi*cChromB/5000*apDef+cSpher*apSpher);
        apR1(isnan(apR1)) = 0;
        apG1(isnan(apG1)) = 0;
        apB1(isnan(apB1)) = 0;
        %imstretch(real(apR1));
        %imstretch(real(apG1));
        %imstretch(real(apB1));
        %imshow(real(apR1));
        %imshow(real(apG1));
        %imshow(real(apB1));
        
        PSF1R = abs(fft2(apR1)).^2;
        PSF1G = abs(fft2(apG1)).^2;
        PSF1B = abs(fft2(apB1)).^2;
        PSF1R = PSF1R/max(max(PSF1R)); %abs(fft2(apR1)).^2;
        PSF1G = PSF1G/max(max(PSF1G));
        PSF1B = PSF1B/max(max(PSF1B));
        %off = 10; 
        
        %PSF1R(fftshift(rad)>.5) = 0.0; %PSF1R(512-off:512+off,512-off:512+off);
        %PSF1G(fftshift(rad)>.5) = 0.0; %\
        %PSF1B(fftshift(rad)>.5) = 0.0;
        
%         PSF1R(PSF1R<1e-2) = 0.0; %PSF1R(512-off:512+off,512-off:512+off);
%         PSF1G(PSF1G<1e-2) = 0.0; %\
%         PSF1B(PSF1B<1e-2) = 0.0; %
        %PSF1G = PSF1G(512-off:512+off,512-off:512+off);
        %PSF1B = PSF1B(512-off:512+off,512-off:512+off);
        %size(PSF1R)
        %PSF1R = zeros(dimX);
        %PSF1R(mid-1:mid+1,mid-1:mid+1) = 1.0;
        %whos
        %figure(); imstretch(fftshift(PSF1R));
        %figure(); mesh(fftshift(PSF1R));
        %imgFTR = fft2(img);
        %size(PSF1R)
        %size(fft2(img(:,:,1)))
        %1+nStk*cChromR
        %figure(); imstretch((conv2(double(img(:,:,1)),PSF1R)));
        %imgStack(:,:,ck) = conv2(double(img(:,:,1)),PSF1R);%;% + conv2(double(img(:,:,2)),PSF1G) +conv2(double(img(:,:,3)),PSF1B);
        %imgStack(:,:,1,ck) = conv2(double(img(:,:,1)),PSF1R); %ifft2(fft2(img(:,:,1)).* (PSF1R));%+ fft2(img(:,:,2)).*PSF1G + fft2(img(:,:,3)).*PSF1B);
        %imgStack(:,:,2,1+nStk*cChromR) = conv2(double(img(:,:,2)),PSF1G);
        %imgStack(:,:,3,1+nStk*cChromR) = conv2(double(img(:,:,3)),PSF1B);
        %test = abs(ifft2(fft2(img(:,:,1)).* (PSF1R)));%+ fft2(img(:,:,2)).*PSF1G + fft2(img(:,:,3)).*PSF1B);
%         imstretch(test)
        
        tempR = abs(ifft2(fft2(img(:,:,1),dimX,dimX).*fft2(PSF1R)));
        tempG = abs(ifft2(fft2(img(:,:,2),dimX,dimX).*fft2(PSF1G)));
        tempB = abs(ifft2(fft2(img(:,:,3),dimX,dimX).*fft2(PSF1B)));
        tempR = (tempR - min(min(tempR)));
        tempR = tempR ./ max(max(tempR));
        tempG = (tempG - min(min(tempG)));
        tempG = tempG ./ max(max(tempG));
        tempB = (tempB - min(min(tempB)));
        tempB = tempB ./ max(max(tempB));
        imgStackRGB(:,:,1,ck) = tempR;
        imgStackRGB(:,:,2,ck) = tempG;
        imgStackRGB(:,:,3,ck) = tempB; 
        imgStack(:,:,ck) = (tempR + tempG + tempB)/3; 
%         imgStack(:,:,3) = ifft2(fft2(img(:,:,3)).*(PSF1B));%+ fft2(img(:,:,2)).*PSF1G + fft2(img(:,:,3)).*PSF1B);
        %conv2(double(img(:,:,1)),PSF1R);
%         max(max(max(apR1)))
%         mean(mean(mean(apG1)))
%         min(min(min(apB1)))
        %1+nStk*cChromR
        %ck
        
        PSF(:,:,1,ck) = PSF1R;
        PSF(:,:,2,1+ck) = PSF1G;
        PSF(:,:,3,1+ck) = PSF1B;
        imgWrt =  (imgStack(:,:,ck));%-min(min(min(imgStack(:,:,:))))) ./ max(max(max(imgStack(:,:,:)-min(min(min(imgStack(:,:,:))))))));
        %imgWrt = imgWrt - m(min(minimgWrt));
        %max(max(imgStack))
        imwrite(imgWrt,strcat('./cuttleFishVision/cuttleSwimming/',fN,'-',int2str(ck),'.jpg'));
    end
 

end
% 
% figure(1); subplot(221); imshow(cutDil1);
% figure(1); subplot(222); imshow(cut);
% subplot(223); imshow(cutEro1);
% subplot(224); imshow(cutEro2);
% 
% win = 15;
%         
% figure(2); subplot(221);  contour(cutDil1PSF(mid-win:mid+win,mid-win:mid+win),'LineWidth',4.5); colormap(hot); view([0 90])
% figure(2); subplot(222);  contour(cutPSF(mid-win:mid+win,mid-win:mid+win),'LineWidth',4.5); colormap(hot); view([0 90])
% figure(2); subplot(223);  contour(cutEro1PSF(mid-win:mid+win,mid-win:mid+win),'LineWidth',4.5); colormap(hot); view([0 90])
% figure(2); subplot(224);  contour(cutEro2PSF(mid-win:mid+win,mid-win:mid+win),'LineWidth',4.5); colormap(hot); view([0 90])
% 
% figure(3); subplot(221);  mesh(cutDil1PSF(mid-win:mid+win,mid-win:mid+win),'LineWidth',4.5); colormap(hot); view([25 65]);
% figure(3); subplot(222);  mesh(cutPSF(mid-win:mid+win,mid-win:mid+win),'LineWidth',4.5); colormap(hot); view([25 65]);
% figure(3); subplot(223);  mesh(cutEro1PSF(mid-win:mid+win,mid-win:mid+win),'LineWidth',4.5); colormap(hot); view([25 65]);
% figure(3); subplot(224);  mesh(cutEro2PSF(mid-win:mid+win,mid-win:mid+win),'LineWidth',4.5); colormap(hot); view([25 65]);
% 
% addpath('/home/qt/Desktop/TIEPhaseScripts');
% figure(4); subplot(221);  imstretch(cutDil1PSF(mid-win:mid+win,mid-win:mid+win)); colormap(hot);% view([25 65]);
% figure(4); subplot(222);  imstretch(cutPSF(mid-win:mid+win,mid-win:mid+win)); colormap(hot); %view([25 65]);
% figure(4); subplot(223);  imstretch(cutEro1PSF(mid-win:mid+win,mid-win:mid+win)); colormap(hot);% view([25 65]);
% figure(4); subplot(224);  imstretch(cutEro2PSF(mid-win:mid+win,mid-win:mid+win)); colormap(hot);% view([25 65]);
% 
% % figure(5); subplot(221);  imstretch(cutDil1FT(mid-win:mid+win,mid-win:mid+win)); colormap(hot);% view([25 65]);
% % figure(5); subplot(222);  imstretch(cutFT(mid-win:mid+win,mid-win:mid+win)); colormap(hot); %view([25 65]);
% % figure(5); subplot(223);  imstretch(cutEro1FT(mid-win:mid+win,mid-win:mid+win)); colormap(hot);% view([25 65]);
% % figure(5); subplot(224);  imstretch(cutEro2FT(mid-win:mid+win,mid-win:mid+win)); colormap(hot);% view([25 65]);
