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
% OUTPUT
%
% dp ........... matrix with the coefficients in Chebyshev basis of the eigenvalues
% vp ........... tensor with the coefficients in Chebyshev basis of the eigenvectors
%

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


[m,n,o]=size(A);
% check size of A
assert(mdA==o);
assert(m==n);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 0th approximation
[V0,D0] = eig(A(:,:,1));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% prepare storange dp, vp, xp, and z
vp = zeros(n,n,md);
dp = zeros(n,md);
vp(:,:,1) = V0;
dp(:,1) = diag(D0);
xp = zeros(n+1,n,md);
z = zeros((n+1)*md,n);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% simultaneous solution for all eigenpairs 

% 0th approximation is already done
xp(1,:,1) = dp(:,1);
xp(2:(n+1),:,1) = vp(:,:,1);

z(1:n+1,:) = xp(:,:,1);


% extend solution to include the kk-th components lambda_kk x_kk 
% we use the Taylor-style block upper triangular igonoring
% that 
%    Un Um = Un+m Un+m-2 ... Un-m is
% and instead assuming it is just
%    Un Um = Un+m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% we solve the lower triangular system by
% forward substitution
for ii = 1:n
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
			F = [0 ctranspose(z(2:(n+1),ii)); z(2:(n+1),ii)  z(1,ii)*eye(n)-A(:,:,1)];
			fprintf('iteration %d Condition %e\n',ii,cond(F));
		end

		if (usesingle)
			F = single(F);
		end

		% record the new, kth, components
		xp(:,ii,kk) = F\[-lp/2;yp];
		dp(ii,kk) = xp(1,ii,kk);
		vp(:,ii,kk) = xp(2:end,ii,kk);
		% copy solution to z
		z(((kk-1)*(n+1))+(1:(n+1)),ii) = xp(:,ii,kk); 
		
	end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% on the final level we do eight Newton steps to improve the solution
for cc = 1:newtonsteps
	z(1:(n+1)*(kk),:) = onestep(A,z(1:(n+1)*(kk),:),n,md);
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
