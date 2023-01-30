function [dp,vp] = cheb_evp(A,md,newtonsteps,mdA,usesingle)
%% Chebyshev approximation 
% INPUT
% 
% A ............ matrix A(mu) given by Chebyshev coefficients
%
% md ........... degree of the approximation + 1
%
% newtonsteps .. (optional) number of step in Newton's method
%
% mdA .......... (optional) degree of the approximation of A 
%
% usesingle .... (optional) force F to be only accurate to single precision
%
% ind_eigenv ... (optional) if set only the eigenvalues (sorted by absolute
%                           value) corresponding to the indizes listed in 
%                           this vector are computed
%
% OUTPUT
%
% dp ........... matrix with the coefficients in Chebyshev basis of the eigenvalues
% vp ........... tensor with the coefficients in Chebyshev basis of the eigenvectors
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialization
[m,n,o]=size(A);
% check size of A
assert(mdA==o);
assert(m==n);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% optional parameters
% number of newtonsteps 
if (~exist('newtonsteps','var'))
	newtonsteps = 8;
end
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
	len = length(ind_eigenv)
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% simultaneous solution for all eigenpairs 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% prepare storange dp, vp, xp, and z
vp = zeros(n,len,md);
dp = zeros(len,md);
xp = zeros(n+1,len,md);
xxp = zeros(n+1,len,md);
z = zeros((n+1)*md,len);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 0th approximation
[U,T] = eig(A(:,:,1));
[V,D0] = eig(T);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% copy 0th approximation 
d0 = diag(D0);
[~,ind] = sort(abs(d0),'descend');

W = U*V;
vp(:,:,1) = W(:,ind(ind_eigenv));
dp(:,1) = d0(ind(ind_eigenv));

xp(1,:,1) = dp(:,1);
xp(2:(n+1),:,1) = vp(:,:,1);

z(1:n+1,:) = xp(:,:,1);
zz(1:n+1,:) = xp(:,:,1);

% extend solution to include the kk-th components lambda_kk x_kk 
% we use the Taylor-style block upper triangular igonoring
% that 
%    Un Um = Un+m Un+m-2 ... Un-m is
% and instead assuming it is just
%    Un Um = Un+m
% This is using a (2.7)-like upper triangular system 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% we solve the lower triangular system by
% forward substitution
for ii = ind_eigenv
	for kk = 2:md

		yp = zeros(n,1);
		lp = zeros(1,1);
		for ll = 1:kk-1
			yp = yp + A(:,:,kk+1-ll)*z(((ll-1)*(n+1))+(2:(n+1)),ii);
			if (ll>1)
				yp = yp - z(((kk-ll)*(n+1))+(2:(n+1)),ii)*z(((ll-1)*(n+1))+1,ii);
				if (ll<kk-1)
					lp = lp + ctranspose(z(((kk-ll)*(n+1))+(2:(n+1)),ii))*z(((ll-1)*(n+1))+(2:(n+1)),ii);
				end
			end
		end
	
	
		% prepare linear system and it's inverse
		if (kk==2)
			% compute ei = U' v_0
			ei = zeros(n,1); ei(ind(ind_eigenv(ii)))=1;

			% old E
			% code lines using the old F are still available as comment
			%F = [0 ctranspose(z(2:(n+1),ii)); z(2:(n+1),ii)  z(1,ii)*eye(n)-A(:,:,1)];

			% new highly efficient E
			E = [0, ctranspose(ei); ei, z(1,ii)*eye(n)-T];

			fprintf('iteration %d Condition %e\n',ii,cond(E));
		end

		if (usesingle)
		  F = single(F);
			E = single(E);
		end

		% old E
		% record the new, kth, components
		%xp(:,ii,kk) = F\[-lp/2;yp];
		%dp(ii,kk) = xp(1,ii,kk);
		%vp(:,ii,kk) = xp(2:end,ii,kk);
		% copy solution to z
		%z(((kk-1)*(n+1))+(1:(n+1)),ii) = xp(:,ii,kk); 

		% new E
		% record the new, kth, components
		xp(:,ii,kk) = E\[-lp/2;U'*yp];
		dp(ii,kk) = xp(1,ii,kk);
		vp(:,ii,kk) = U*xp(2:end,ii,kk);
		xp(2:end,ii,kk) = vp(:,ii,kk);
		% copy solution to z
		z(((kk-1)*(n+1))+(1:(n+1)),ii) = xp(:,ii,kk); 

	end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% on the final level we do eight Newton steps to improve the solution
% now we actually solve (3.4)
for cc = 1:newtonsteps
	z(1:(n+1)*(kk),:) = onestep(A,z(1:(n+1)*(kk),:));%,n,md);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% copy solution to xp, dp, and vp
for ii=1:n		
	for kk = 1:md
		xp(:,ii,kk) = z(((kk-1)*(n+1))+(1:(n+1)),ii);
		dp(ii,kk) = xp(1,ii,kk);
		vp(:,ii,kk) = xp(2:end,ii,kk);
	end
end





%%% Local Variables: 
%%% mode:matlab
%%% flyspell-mode:nil
%%% mode:flyspell-pro
