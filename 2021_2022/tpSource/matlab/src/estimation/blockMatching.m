function [u, v] = blockMatching( img1, img2, options )
% Estimation du mouvement par block matching
% Inputs:
%     - img1: image a t
%     - img2: image a t+1
%     - options: structure contenant des arguments supplementaires (optionel)
%         Les champs possibles (valeur par defaut) sont:
%         - dimB (2): 1/2 taille du bloc
%         - dimR (5): 1/2 hauteur de la zone de recherche
% Outputs:
%     - u, v: deplacement suivant x et y
% 
% Author: Thomas Dietenbeck
    dOptions = struct( 'dimB', 2, 'dimR', 5 );
    %-- Process inputs
    if( ~exist( 'options', 'var' ) )
        options = dOptions;
    elseif( ~isstruct( options ) )
        error( 'BlockMatching:Error', 'Options must be a struct' );
    else
        tags = fieldnames( dOptions );
        idxT = find( ~isfield( options, tags ) )';
        for i = idxT
            options.(tags{i}) = dOptions.(tags{i});
        end
    end
    dimB = options.dimB;
    dimR = options.dimR;
    
    
    sImg = size( img1 );
    sX = sImg(1);           sY = sImg(2);
    u = zeros( sImg );      v = zeros( sImg );
    for x = dimB+1:sX-dimB     % Pour avoir la place du bloc
        for y = dimB+1:sY-dimB
            subR = img1( x-dimB:x+dimB, y-dimB:y+dimB );
            dMin = flintmax;
%             subI = img2( x-dimB:x+dimB, y-dimB:y+dimB );
%             dMin = getDistance( subR, subI );
            uTmp = 0;       vTmp = 0;
            for dX = -dimR:1:+dimR
                xP = x + dX;
                if( ((xP - dimB) >= 1) && ((xP + dimB) <= sX) )
                    for dY = -dimR:1:dimR
                        yP = y + dY;
                        if( ((yP - dimB) >= 1) && ((yP + dimB) <= sY) )
                            subI = img2( xP-dimB:xP+dimB, yP-dimB:yP+dimB );
                            d = getDistance( subR, subI );
                            if( d < dMin )
                                dMin = d;
                                uTmp = dX;
                                vTmp = dY;
                            end
                        end
                    end
                end
            end
            u(x, y) = uTmp;     v(x, y) = vTmp;
        end
    end
    
    
    function d = getDistance( subR, subI )
        d = sum( abs( subR(:) - subI(:) ) );	% MAD
%         d = sum( ( subR(:) - subI(:) ).^2 );    % MSSD
        