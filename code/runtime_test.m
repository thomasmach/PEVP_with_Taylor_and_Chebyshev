%% runtime test
% this file generates the data in Table 2.2

% outputting warnings is time consuming, thus we turn them off here
warning('off','all')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% we need chebfun
% add the correct folder if needed
addpath('~/git/chebfun')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% parameters 

% we use double precision Taylor
usevpa = false;
chebyshev = false;
usesingle = false;
taylor = true;
% Example 2.3 from the paper
example = 1;
% parameter expansion point (Taylor approximation)
t0 = 0.2;		
nrc = 25;
% points
r1 = 5; % big radius torus
r2 = 1; % small radius torus

% windings
if (~exist('k','var'))
	k = 2; 
end



NN = [8,16,32,64,128,256,512,1024,2048,4096,8,16,32,64,128,256,512,1024,2048,4096,8,8,8,8,8,8,8,8,8,8];
PP = [2,2,2,2,2,2,2,2,2,2,7,7,7,7,7,7,7,7,7,7,2,4,8,16,32,64,1,1,1,1];
lNN = length(NN)

for in = 1:lNN
	
	n = NN(in);
	p = PP(in);
	pA = p;

	rn = linspace(0,1,n+1);
	rn = rn(1:n);

	% points on the windings 
	P = zeros(n,3);
	P(:,1) = cos(2*pi*rn).*(r1+r2*cos(2*pi*k*rn));
	P(:,2) = sin(2*pi*rn).*(r1+r2*cos(2*pi*k*rn));
	P(:,3) = r2*sin(2*pi*k*rn);
	n_to_make_P = n;

	% setup points
	U = zeros(n,n);
	for ii = 1:n
		for jj = 1:n
			U(ii,jj) = sqrt((P(ii,:)-P(jj,:))*transpose(P(ii,:)-P(jj,:)));
		end
	end

	% setup matrix
	tic

	A = zeros(n,n,p);
	for kk = 1:p
		A(:,:,kk) = (-U).^(kk-1).*exp(-t0*U);
	end
			
	time_generating_or_computing_A = toc

	if (n<1000)
		nrccurrent = nrc;
	else
		if (n<2000)
			nrccurrent = 2;
		else
			nrccurrent = 1;
		end
	end
	
	% compute Taylor approximation
	tic
	
	for rc=1:nrccurrent
		[dp,vp] = taylor_evp(A,p,pA,usesingle);
	end
	
	time_taylor = toc

	time(in) = time_taylor/nrccurrent;
	

end

filename = sprintf('exp_pap_runtime_table.tex');
tout = fopen(filename,'w');

cl = lNN/3;
for ii=1:cl
	if (ii==1)
		fprintf(tout,'%d & %d & %8.4f & ---&\n',NN(ii),PP(ii),time(ii));
		fprintf(tout,'%d & %d & %8.4f & ---&\n',NN(cl+ii),PP(cl+ii),time(cl+ii));
		fprintf(tout,'%d & %d & %8.4f & ---\\\\\n',NN(2*cl+ii),PP(2*cl+ii),time(2*cl+ii));
	else
		fprintf(tout,'%d & %d & %8.4f & %6.2f &\n',NN(ii),PP(ii),time(ii),time(ii)/time(ii-1));
		fprintf(tout,'%d & %d & %8.4f & %6.2f &\n',NN(cl+ii),PP(cl+ii),time(cl+ii),time(cl+ii)/time(cl+ii-1));
		fprintf(tout,'%d & %d & %8.4f & %6.2f \\\\\n',NN(2*cl+ii),PP(2*cl+ii),time(2*cl+ii),time(2*cl+ii)/time(2*cl+ii-1));
	end
end
fclose(tout)

warning('on','all')





















%%% Local Variables: 
%%% mode:matlab
%%% flyspell-mode:nil
%%% mode:flyspell-prog
