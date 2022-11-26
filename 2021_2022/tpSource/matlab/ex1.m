clear variables
close all
clc

addpath( genpath( 'src' ) )

dataPath = '../../data/TP/';
seqNum = 1;
switch seqNum
    case 1
        seqName = 'metro/';
        refName = 'refMetro.png';
    case 2
        seqName = 'hall/';
        refName = 'refHall.png';
    case 3
        seqName = 'Coke/';
        refName = 'refCoke.png';
    case 4
        seqName = 'jump/';
        refName = 'refJump.png';
end
seq = double( readSeq( [dataPath, seqName], 1 ) );
refImg = double( imread( [dataPath, refName] ) );
sSeq = size( seq );

%% Parametres
typeDetect = 1;     % 1 = difference d'images, 2 = ElGammal, 3 = ACP
tDeb = 1;           % Frame de debut de la detection
tFin = sSeq(3);     % Frame de fin de la detection
    %-- Difference d'images
typeDiff = 1;       % Methode de difference: 1 = fond fixe, 2 = 2 images, 3 = fond connu avec mise a jour
tStep = 1;          % Difference de temps entre 2 images successives
alpha = 0.05;       % Facteur d'apprentissage (si typeDiff = 3)
optionsDI = struct( 'tDeb', tDeb, 'tFin', tFin, 'type', typeDiff, 'refImg', refImg, 'tStep', tStep, 'alpha', alpha );
    %-- ElGammal
nEG = 50;           % Nombre de frames utilisees pour l'apprentissage
typeEG = 1;
nMaxEG = sSeq(3);
optionsEG = struct( 'tDeb', tDeb, 'tFin', tFin, 'N', nEG, 'type', typeEG, 'NMax', nMaxEG );
    %-- ACP
nACP = 50;          % Nombre de frames utilisees pour l'apprentissage
M = 0.70;           % Pourcentage d'eigenbackgrounds
typeACP = 1;
nMaxACP = sSeq(3);
optionsACP = struct( 'tDeb', tDeb, 'tFin', tFin, 'N', nACP, 'type', typeACP, 'NMax', nMaxACP, 'M', M );

%% Detection
switch typeDetect
    case 1
		seqD = bgDiffImg( seq, optionsDI );
    case 2          % Elgammal
        seqD = bgElgammal( seq, optionsEG );
    case 3          % ACP sur sequence
		seqD = bgACP( seq, optionsACP );
end

%% Display
figure;
for t = tDeb:1:tFin
    subplot(1, 2, 1); imagesc( seq(:,:,t) ); axis image; axis off; colormap gray;
    subplot(1, 2, 2); imagesc( seqD(:,:,t) ); axis image; axis off; colormap gray; caxis( [0, 1] );
    title( [ num2str(t), ' / ', num2str( sSeq(3) ) ] );
    pause(0.1);
end