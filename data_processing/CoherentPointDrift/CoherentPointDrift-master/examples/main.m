% Example 3. 3D Rigid CPD point-set registration. Full options intialization.
%  3D face point-set.
clear all; close all; clc;

in_file = 'orig.aln';
out_file = 'trans.aln';

% load cpd_data3D_face.mat;%Y=X;

% [src, t] = read_ply('X1.ply','');
% [dst, t] = read_ply('Y1.ply','');

[X, Y_, mat1, mat2, f1, f2] = load_mesh(in_file);

X = transformM(X, mat1);
Y = transformM(Y_, mat2);

% disp(['res1.ndims: ' num2str(ndims(res1))]);
% disp(['res1: ' res1]);
% [res2, t] = read_ply('Y1.ply','');

[x, y] = size(X);
% sprintf('X.dimention: %d, %d\n',x, y);
% disp(['X.dimention: ' num2str(x) ' ' num2str(y)]);

% add a random rigid transformation
% R=cpd_R(rand(1),rand(1),rand(1));
% X=rand(1)*X*R'+1;

export_ply3d('X_orig.ply', X)
export_ply3d('Y_orig.ply', Y)

% Set the options
opt.method='rigid'; % use rigid registration
opt.viz=1;          % show every iteration
opt.outliers=0.1;     % do not assume any noise 

opt.normalize=0;    % normalize to unit variance and zero mean before registering (default)
opt.scale=1;        % estimate global scaling too (default)
opt.rot=1;          % estimate strictly rotational matrix (default)
opt.corresp=0;      % do not compute the correspondence vector at the end of registration (default)

opt.max_it=80;     % max number of iterations
opt.tol=1e-5;       % tolerance


% registering Y to X
[Transform, Correspondence]=cpd_register(X,Y,opt);

figure,cpd_plot_iter(X, Y); title('Before');
figure,cpd_plot_iter(X, Transform.Y);  title('After registering Y to X');

export_ply3d('fittedY.ply', Transform.Y)

transformedY = transformRTS(Y, Transform.R, Transform.t, Transform.s);
% export_ply3d('transformedY.ply', transformedY)

M = updateTransformMat(mat2, Transform.R, Transform.t, Transform.s);
transformedY2 = transformM(Y_, M);
export_ply3d('transformedY.ply', transformedY2);

save_aln(out_file, f1, f2, mat1, M);


function save_aln(out_file, f1, f2, mat1, fittedm)
    disp(['write: ' out_file]);
    fp=fopen(out_file, 'w');
    fprintf(fp, '%d\n', 2);
    fprintf(fp, '%s\n', f1);
    fprintf(fp, '#\n');
    for i = 1:4
        fprintf(fp, '%f %f %f %f\n', mat1(1,i), mat1(2,i), mat1(3,i), mat1(4,i));
    end
%     fprintf(fp, '%s\n', f2);
%     fprintf(fp, '#\n');
%     for i = 1:4
%         fprintf(fp, '%f %f %f %f\n', mat2(1,i), mat2(2,i), mat2(3,i), mat2(4,i));
%     end
    fprintf(fp, '%s\n', f2);
    fprintf(fp, '#\n');
    for i = 1:4
        fprintf(fp, '%f %f %f %f\n', fittedm(1,i), fittedm(2,i), fittedm(3,i), fittedm(4,i));
    end
    fclose(fp);
end

function X = transformM(X1, mat)
    s = mat(4, 4)
    r = mat(1:3,1:3)
    t = mat(4,1:3)

    X = s * X1 * r + t;
end

function X = transformRTS(X1, R, T, S)
%     X = S * X1 * R;
%     X = X + T';
%     X = X * S;
    [M, D]=size(X1);
    X=S*X1*R'+T';%repmat(T',[M 1]);
end

function M = updateTransformMat(M1, r2, t2, s2)
    s1 = M1(4, 4)
    r1 = M1(1:3,1:3)
    t1 = M1(4,1:3)

%     X = s2 * (s1 * X1 * r1 + t1) *r2' + t2';
    M = zeros(4, 4);
    M(1:3,1:3) = s1 * s2 * r1 * r2';
    M(4, 4) = 1.0;
    M(4, 1:3) = s1 * s2 * t1 * r2' + t2';
    M
end

function [X, Y, mat1, mat2, f1, f2] = load_mesh(aln_file)
    fp=fopen(aln_file, 'r');
    file_count = fscanf(fp,'%d',1);
    f1 = fscanf(fp,'%s',1);
    disp(['f1: ', f1]);
    t = fscanf(fp,'%s',1);
    [src, t] = read_ply(f1,'');
    mat1 = fscanf(fp,'%f %f',[4,4]);
    disp(['mat1: ', f1]);
    f2 = fscanf(fp,'%s',1);
    t = fscanf(fp,'%s',1);  
    disp(['f2: ', f2]);
    mat2 = fscanf(fp,'%f %f',[4,4]);
    [dst, t] = read_ply(f2,'');  
    fclose(fp);
    
    len1 = length(src.vertex.x);
    disp(['X.len: ', num2str(len1)]);
    X = zeros(len1, 3);
    for i = 1:len1
        X(i,1) = src.vertex.x(i);
        X(i,2) = src.vertex.y(i);
        X(i,3) = src.vertex.z(i);
    end

    len2 = length(dst.vertex.x);
    disp(['Y.len: ', num2str(len2)]);
    Y = zeros(len2, 3);
    for i = 1:len2
        Y(i,1) = dst.vertex.x(i);
        Y(i,2) = dst.vertex.y(i);
        Y(i,3) = dst.vertex.z(i);
    end
end
function export_ply3d(filename, mesh)
    disp(filename);
    fp=fopen(filename, 'w');
    [count, tmp] = size(mesh);
%     sprintf('tilt.dimension: %d %d\n',xt, yt);
    fprintf(fp, 'ply\n');
	fprintf(fp, 'format ascii 1.0\n');
	fprintf(fp, 'comment (C) Hongzhi Wu, Sep 2013.\n');

	fprintf(fp, 'element vertex %d\n', count);
	fprintf(fp, 'property float x\n');
	fprintf(fp, 'property float y\n');
	fprintf(fp, 'property float z\n');
    fprintf(fp, 'end_header\n');
    for i = 1:count
        fprintf(fp, '%f %f %f\n', mesh(i,1), mesh(i,2), mesh(i,3));
    end
    
    fclose(fp);
end
