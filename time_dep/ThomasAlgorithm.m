function s=ThomasAlgorithm(a,b,c,f,n)
%-----solve 3-diagonal system--------

d=zeros(n,1);
u=zeros(n,1);
s = zeros(n,1);
for i=1:n-1
    if i==1
       u(i)=c(i)/b(i);
    else
    u(i)=c(i)/(b(i)-a(i-1)*u(i-1));  
    end
end

for i = 1:n
    if i==1
        d(i) = f(i)/b(i);
    else
        d(i) = (f(i) - a(i-1)*d(i-1))/(b(i) - a(i-1)*u(i-1));
    end
end
s(n) = d(n);
for i=n-1:-1:1
    s(i)=d(i)-u(i)*s(i+1);
end
% %%%%%%%example using Thomas Algorithm
% A = [1 1 0; 0 1 1; 0 0 1]';
% f= [1;2;3];
% a = [1;1]; \lower
% b = [1;1;1];
% c = [0;0]; %upper
% n = 3;
% s = ThomasAlgorithm(a,b,c,f,n);
% sol = trim(A,f);