%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load matrix
load('A1000.mat');
% check size
[m,n,o] = size(A);
assert(m==n);
mdA = o; 
md = o;

% size of Krylov subspace
k = 50;

% which eigenvalues do we care about
ind_eigenv = 1:15

% % short link to A0
% A0 = A(:,:,1);
% 
% 
% [H,V] = arnoldi(A0,k);
% [Uh,Th] = schur(H(1:k,1:k));
% [Vh,D0h] = eig(Th);
% 
% d0h = diag(D0h);
% [~,indh] = sort(abs(d0h),'descend');
% 
% Wh = V(:,1:k)*Uh*Vh;
% vph(:,:,1) = Wh(:,indh(ind_eigenv));
% dph(:,1) = d0h(indh(ind_eigenv));

usesingle = false;
p = 26;
pA = 26;

tic,[dp1,vp1] = taylor_evp(A,p,pA,usesingle,ind_eigenv);toc
tic,[dp2,vp2] = taylor_evp2(A,k,p,pA,usesingle,ind_eigenv);toc
tic,[dp3,vp3] = taylor_evp3(A,k,p,pA,usesingle,ind_eigenv);toc


%%% Local Variables: 
%%% mode:matlab
%%% flyspell-mode:nil
%%% mode:flyspell-prog
%%% ispell-local-dictionary: "american"
%%% End: 
