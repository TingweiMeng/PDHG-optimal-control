% dL is the grad of Lagrangian, it takes two var x, a and returns proj_{a+dL(x)}(0) - a
%       so, when L is differentiable, it simply ignores the 2nd argument and returns dL(x)
function [H, dH, J, alpha, dL] = generating_H_J(Hind, Jind, x_period)
if Hind == 1 % transport
    H = @(p) p;
    dH = @(p) 0.*p+1;
    dL = @(x,a) -a;  % because L is the indicator of 1, and we assume x = 1
else
    if Hind == 2 % Burgers
        H = @(p) p.^2/2;
        dH = @(p) p;
        dL = @(x,a) x;  % L = |.|^2/2
    else
        if Hind == 3 % L1
            H = @(p) abs(p);
            dH = @(p) sign(p);
            dL = @(x,a) dL_ind_negone2one(x, a);
        end
    end
end

if Jind == 1
    % example 1: zero initial condition
    J = @(x) 0.*x;
    alpha = 0;
else
    if Jind == 2
        % example 2: sin initial condition
        alpha = 2*pi / x_period;
        J = @(x) sin(alpha * x);
    end
end

end



% the derivative of indicator fn of [-1,1]
% takes input x,a and returns proj_{a + dL(x)}(x)
function dL = dL_ind_negone2one(x, a)
dL = 0.*x;   % if x in (-1,1), returns 0
eps = 1e-6;
% if x = -1, returns proj_{(-infty, a]}(0) - a
%       eqts -a if a>=0; 0 if a < 0
% if x = 1, returns proj_{[a, infty)}(0) - a
%       eqts -a if a<=0; 0 if a > 0
ind = ((x <= -1 + eps) & (a >= 0)) | ((x >= 1 - eps) & (a <= 0));
dL(ind) = -a(ind);
end