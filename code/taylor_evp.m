function [dp,vp] = taylor_evp(A,md,mdA,usesingle)
%% Taylor approximation 
% INPUT
% 
% A ......... matrix A(mu) given by Taylor coefficients 
%
% md ........ degree of the approximation + 1
%
% mdA ....... (optional) degree of the approximation of A 
%
% OUTPUT
%
% dp ........ matrix with the coefficients in Chebyshev basis of the eigenvalues
% vp ........ tensor with the coefficients in Chebyshev basis of the eigenvectors
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% optional parameters
% max degree A 
if (~exist('mdA','var'))
	mdA = md;
end
% usesingle
if (~exist('usesingle','var'))
	usesingle = false;
end


[m,n,o]=size(A);
% check size of A
assert(mdA==o);
assert(m==n);
% compute the dominant len eigenvalues
if (~exist('len','var'))
	len = n;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% simultaneous solution for all eigenpairs 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 0th approximation
%[V,D0] = eigs(A(:,:,1),len,'largestreal');
[V,D0] = eig(A(:,:,1));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d0 = diag(D0);
[~,ind] = sort(d0,'descend');

vp = zeros(n,len,md);
dp = zeros(len,md);
vp(:,:,1) = V(:,ind(1:len));
dp(:,1) = d0(ind(1:len));
xp = zeros(n+1,len,md);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% we solve the lower triangular system by
% forward substitution
for ii = 1:len
	% ii
	F = [0, ctranspose(vp(:,ii,1)); vp(:,ii,1), dp(ii,1)*eye(n)-A(:,:,1)];

 	if (usesingle)
		F = single(F);
	end
	
	invF = pinv(F);
	
	for kk = 2:md
		yp = zeros(n,1);
		lp = zeros(1,1);
		for ll = 1:kk-1
			yp = yp + nchoosek(kk-1,ll-1)*A(:,:,kk+1-ll)*vp(:,ii,ll);
			if (ll>1)
				yp = yp - nchoosek(kk-1,ll-1)*vp(:,ii,kk+1-ll)*diag(dp(ii,ll));
				if (ll<kk-1)
					lp = lp + diag(nchoosek(kk-1,ll-1)*vp(:,ii,kk+1-ll)'*vp(:,ii,ll));
				end
			end
		end
		

		xp(:,ii,kk) = invF*[-lp/2;yp];
		dp(ii,kk) = xp(1,ii,kk);
		vp(:,ii,kk) = xp(2:end,ii,kk);
		
	end
	
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% we are done





%%% Local Variables: 
%%% mode:matlab
%%% flyspell-mode:nil
%%% mode:flyspell-prog
