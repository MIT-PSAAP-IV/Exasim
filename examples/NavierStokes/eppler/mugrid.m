
alpha = 3;
n = 17;
mu1=logdec(linspace(1000,10000,n), alpha);
mu2=logdec(linspace(0,10,n), alpha);
[mu1, mu2] = meshgrid(mu1, mu2);

figure(1); clf; plot(mu1, mu2, '*b');

alpha = 3;
n = 9;
mu1=logdec(linspace(1000,10000,n), alpha);
mu2=logdec(linspace(0,10,n), alpha);
[mu1, mu2] = meshgrid(mu1, mu2);

figure(2); clf; plot(mu1, mu2, '*b');
