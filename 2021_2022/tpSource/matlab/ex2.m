clear variables
close all
clc

addpath( genpath( 'src' ) )

%% Choix de l'image
isReal = 0;
if( isReal )    % Image reelle
    img1 = imread( '../../../data/girl1.png' );
    img2 = imread( '../../../data/girl2.png' );
    img1 = double( rgb2gray( img1 ) );
    img2 = double( rgb2gray( img2 ) );
    subS = 3;
else    % Image test (binaire)
    vBlock = [4, -2];    % Deplacement (x, y) du carre
    isNoise = 0;        % Images bruitees ou pas
    isIntensity = 0;    % Changement d'illumination ou pas
    [img1, img2] = getImgsBW( vBlock, isNoise, isIntensity );
    subS = 1;
end

%% Parametres
typeEstim = 1;  % 1 = block matching, 2 = 4 Step Search, 3 = Lucas & Kanade, 4 = Horn & Schunck, 5 = Bruhn
sImg = size( img1 );    % Taille de l'image
    %-- Block matching
dimB = 2;       % 1/2 taille du bloc
dimR = 5;       % 1/2 taille de la zone de recherche
N = 1;          % Nombre d'estimations / compensations
optionsBM = struct( 'dimB', dimB, 'dimR', dimR );
    %-- 4 Step Search
options4SS = struct( 'dimB', dimB );
    %-- Lucas-Kanade & Bruhn
sW = 7;        % 1/2 largeur de la fenetre de ponderation
typeW = 0;      % Type de la fenetre de ponderation (0 = uniforme, 1 = gaussienne)
optionsLK = struct( 'sW', sW, 'typeW', typeW );
    %-- Horn-Schunck & Bruhn
alpha = 1;      % Regularisation du champ
maxIts = 250;   % Nombre d'iteration
tol = 1e-5;     % Tolerance entre 2 iterations
optionsHS = struct( 'alpha', alpha, 'maxIts', maxIts, 'tol', tol );
    %-- Bruhn
sW = 5;        % 1/2 largeur de la fenetre de ponderation
optionsBruhn = struct( 'alpha', alpha, 'maxIts', maxIts, 'tol', tol, 'sW', sW );
clear dimB dimR sW typeW alpha maxIts tol;

%% Estimation
tic;
switch typeEstim
    case 1         %-- Block matching
        [v, u] = blockMatching( img1, img2, optionsBM );
    case 2         %-- 4 Step Search
        [u, v] = bm4SS( img1, img2, options4SS );
    case 3         %-- Lucas - Kanade
        [u, v] = ofLK( img1, img2, optionsLK );
    case 4         %-- Horn - Schunck
        [u, v] = ofHS( img1, img2, optionsHS );
    case 5         %-- Bruhn
        [u, v] = ofBruhn( img1, img2, optionsBruhn );
end
toc;


%% Display
[X, Y] = meshgrid( 1:sImg(2), 1:sImg(1) );
figure; imagesc( img1 ); axis image; axis off; colormap gray;
hold on;
    if( ~isReal )
        contour( img2, [0.5, 0.5], 'b', 'linewidth', 2 );
    end
    quiver( X(1:subS:end, 1:subS:end), Y(1:subS:end, 1:subS:end), u(1:subS:end, 1:subS:end), v(1:subS:end, 1:subS:end), 2, 'r' );
hold off;
