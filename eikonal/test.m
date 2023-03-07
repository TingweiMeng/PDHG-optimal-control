
%% one dimentional
dim = 1;

% domain [-1,1]
f = [1; 0];
n_grid = 51; %101;
dx = 2.0 / (n_grid -1);
% phi0 = (1.0: dx: 3.0)';
% phi0 = min((1.0: dx: 3.0)', (3.0: -dx: 1.0)');
phi0 = rand(n_grid,1);

phi = pdhg_eikonal_abs_onedim(f, phi0, dx);

figure; plot(-1:dx:1, phi); title('phi');


%% two dimensional
% domain [-1,1] \times [-1,1]
% phi(-1) = 1; phi(1) = 3
nx = 51; ny = 61;
fx = zeros(nx, 2);
fy = zeros(ny, 2);
dy = 2.0 / (ny-1);
dx = 2.0 / (nx -1);
% phi0 = (1.0: dx: 3.0)';
% phi0 = min((1.0: dx: 3.0)', (3.0: -dx: 1.0)');
x = repmat((-1:dx:1)', [1, ny]);
y = repmat(-1:dy:1, [nx, 1]);
% 2-norm
% phi0 = sqrt(min(x+1, 1-x).^2 + min(y+1, 1-y).^2);
% 1-norm
phi0 = min(min(x+1, 1-x), min(y+1, 1-y));
% phi0 = rand(nx,ny);

phi = pdhg_eikonal_abs_twodim(fx, fy, phi0, dx, dy);
figure; surf(x, y, phi); title('phi');