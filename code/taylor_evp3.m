function [dp,vp] = taylor_evp3(A,k,md,mdA,usesingle,ind_eigenv)
%% Taylor approximation 
% INPUT
% 
% A ........... matrix A(mu) given by Taylor coefficients 
%
% k ........... dimension Krylov subspaces
%
% md .......... degree of the approximation + 1
%
% mdA ......... (optional) degree of the approximation of A 
%
% usesingle ... (optional) if true, then matrix E gets rounded to single precision
%
% ind_eigenv .. (optional) if set only the eigenvalues (sorted by absolute
%                          value) corresponding to the indizes listed in 
%                          this vector are computed
%
% OUTPUT
%
% dp ........ matrix with the Taylor coefficients of the eigenvalues
% vp ........ tensor with the Taylor coefficients of the eigenvectors
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialization
[m,n,o]=size(A);
% check size of A
assert(mdA==o);
assert(m==n);


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
% set ind_eigenv if not set
if (~exist('ind_eigenv','var'))
	ind_eigenv = 1:n;
	len = length(ind_eigenv);
else
	len = length(ind_eigenv);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% simultaneous solution for all eigenpairs 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% prepare storange dp, vp, and xp
vp = zeros(n,len,md);
dp = zeros(len,md);
xp = zeros(n+1,len,md);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 0th approximation
[H,Va] = arnoldi(A(:,:,1),k);
[U,T] = schur(H(1:k,1:k));
[V,D0] = eig(T);
% if A is sparse and only some eigenpairs are of
% interested use for instance 
% [V,D0] = eigs(A(:,:,1),len,'largestreal');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% copy 0th approximation 
d0 = diag(D0);
[~,ind] = sort(abs(d0),'descend');

W = Va(:,1:k)*U*V;
vp(:,:,1) = W(:,ind(ind_eigenv));
dp(:,1) = d0(ind(ind_eigenv));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% we solve the lower triangular system by
% forward substitution
for ii = 1:len

	% compute ei = U' v_0
	% ei = zeros(n,1); ei(ind(ind_eigenv(ii)))=1; 
	ei = Va(:,1:k)*U'*(ctranspose(Va(:,1:k))*vp(:,ii,1));

	% old E
	% code lines using the old F are still available as comment
	F = [0, ctranspose(vp(:,ii,1)); vp(:,ii,1), dp(ii,1)*eye(n)-A(:,:,1)];

	% new highly efficient E
	% E = [0, ctranspose(ei); ei, dp(ii,1)*eye(n)-Va(:,1:k)*T*(ctranspose(Va(:,1:k)))];

 	if (usesingle)
		F = single(F);
		% E = single(E);
	end
		
	% invF = pinv(F);
	
	% compute right hand side of (2.5)
	for kk = 2:md
		yp = zeros(n,1);
		lp = zeros(1,1);
		for ll = 1:kk-1
			% yp = yp + nchoosek(kk-1,ll-1)*A(:,:,kk+1-ll)*vp(:,ii,ll);
			yp = yp + A(:,:,kk+1-ll)*vp(:,ii,ll);
			if (ll>1)
				% yp = yp - nchoosek(kk-1,ll-1)*vp(:,ii,kk+1-ll)*diag(dp(ii,ll));
				yp = yp - vp(:,ii,kk+1-ll)*diag(dp(ii,ll));
				if (ll<kk-1)
					%	lp = lp + diag(nchoosek(kk-1,ll-1)*vp(:,ii,kk+1-ll)'*vp(:,ii,ll));
					lp = lp + diag(vp(:,ii,kk+1-ll)'*vp(:,ii,ll));
				end
			end
		end
		
		% solve (2.5)

		% xp(:,ii,kk) = invF*[-lp/2;yp];
		% dp(ii,kk) = xp(1,ii,kk);
		% vp(:,ii,kk) = xp(2:end,ii,kk);
		
		% solve (2.5) using a reformulation based on (2.6)
		%auxvec = [-lp/2;Va(:,1:k)*ctranspose(U)*ctranspose(Va(:,1:k))*yp];
		auxvec = [-lp/2;yp];
		[H2,V2] = arnoldi(F,k,auxvec);
		xp(:,ii,kk) = norm(auxvec)*V2(:,1:k) * (H2\(eye(k+1,1)));
		%xp(:,ii,kk) = E\[-lp/2;U'*yp];
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
