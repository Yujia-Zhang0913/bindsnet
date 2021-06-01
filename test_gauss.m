t = 0:0.0001:1;
n=10;
omiga = 1/n/3;
for i=0:n
   y = exp(-(1/n*i-t).^2/2/omiga^2);
   plot(t,y);
   hold on
end


