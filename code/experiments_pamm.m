%% numerical experiments 
%% generating a test matrix

% we need chebfun
% add the correct folder if needed
addpath('~/git/chebfun')


% type of experiments, the ones for the paper
type = 'pap';

% selects different sets of parameters for different  numerical experiments
if (~exist('select','var'))
	select = 1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Table Select
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% select   floats            filenames      method                   example
% 1          Fig 3.1           fig31          Chebyshev               5 (matlab) 1 (paper)
%  

rng('default');


usevpa = false;
chebyshev = false;
isclose = false;
npoints = 151;
realizations = 1;
timesteps = 50;  % number of timesteps

% parameter interval (Chebyshev approximation)
T2 = [2, 4];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% floats 
% 1  figures eigenvalues sampled and Taylor approximated, magnification
% 2  table sum (not used)
% 3  figures difference eigenvalues and approximation
% 4  like 3 but for Chebyshev
% 5  like 1 but example 2
% 6  like 3 but example 3
% 7  like 5 but Chebyshev
% 8  like 6 but Chebyshev
% 9  like 1 but example 3 mu = -0.2
% 10 like 1 but example 3 mu = 0.2
% 11 figure of guessing mu with ksdensity
switch select
	case {1}
		% eigenvalues vs approximation
		floats = [1];

		n = 20;
		n2= 10;
		p = 5;
		% parameter interval (Chebyshev approximation)
		T2 = [2, 4];
		% parameter interval (plotting)
		T = [1, 5];
		% parameter figure magnifying
		S = [42,42,42,42]; % turn off magnification

		usevpa = false;
		usesingle = false;

		taylor = false;
		chebyshev = true;
		
		% select example
		example = 5;
		
		% no of points for plot
		npoints = 201;
		
		% no of newtonsteps
		newtonsteps = 3;

		% step size discrete system
		h = 1/10;
		
		% hidden parameter to be uncovered
		mu_hidden = 3.2;

		noise_eps = 1e-5;

	case {2}
		% eigenvalues vs approximation
		floats = [3,21];

		n = 20;
		n2= 10;
		p = 40;
		% parameter interval (Chebyshev approximation)
		T2 = [2, 4];
		% parameter interval (plotting)
		T = [1, 5];
		% parameter figure magnifying
		S = [42,42,42,42]; % turn off magnification

		usevpa = false;
		usesingle = false;

		taylor = false;
		chebyshev = true;
		
		% select example
		example = 5;
		
		% no of points for plot
		npoints = 201;
		
		% no of newtonsteps
		newtonsteps = 8;

		% step size discrete system
		h = 1/10;
		
		% hidden parameter to be uncovered
		mu_hidden = 3.2;

		% for plots
		t0 = 3;

		noise_eps = 1e-5;

	case {3}
		% parameter estimation start solution far away from equilibrium
		floats = [11,21];

		n = 20;
		n2= 10;
		p = 10;
		% parameter interval (Chebyshev approximation)
		T2 = [2, 4];
		% parameter interval (plotting)
		T = [1, 5];
		% parameter figure magnifying
		S = [42,42,42,42]; % turn off magnification

		usevpa = false;
		usesingle = false;

		taylor = false;
		chebyshev = true;
		
		% select example
		example = 5;
		
		% no of points for plot
		npoints = 201;
		
		% no of newtonsteps
		newtonsteps = 8;

		% step size discrete system
		h = 1/10;
		
		% hidden parameter to be uncovered
		mu_hidden = 3.2;

		% for plots
		t0 = 3;
		
		noise_eps = 1e-5;
		
	case {4}
		% parameter estimation for start solution close to equilibrium
		floats = [11];

		n = 20;
		n2= 10;
		p = 10;
		% parameter interval (Chebyshev approximation)
		T2 = [2, 4];
		% parameter interval (plotting)
		T = [1, 5];
		% parameter figure magnifying
		S = [42,42,42,42]; % turn off magnification

		usevpa = false;
		usesingle = false;

		taylor = false;
		chebyshev = true;
		isclose = true
		
		% select example
		example = 5;
		
		% no of points for plot
		npoints = 201;
		
		% no of newtonsteps
		newtonsteps = 8;

		% step size discrete system
		h = 1/10;
		
		% hidden parameter to be uncovered
		mu_hidden = 3.2;

		% for plots
		t0 = 3;
		
		noise_eps = 1e-5;
		
		realizations = 100;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% output on
if (~exist('output','var'))
	output = true;
end

% size of points, dimension of A
if (~exist('n','var'))
	n = 8;
end

% newtonsteps
if (~exist('newtonsteps','var'))
	newtonsteps = 20;
end

% 0, 1, and 2 in as Chebfuns
null = chebfun(@(t) 0, T2, 'vectorize','splitting','on');
eins = chebfun(@(t) 1, T2, 'vectorize','splitting','on');
zwei = chebfun(@(t) 2, T2, 'vectorize','splitting','on');

% generating Chebyshev polynomials Uscc{ii} of second kind in Chebfun
gen_tsch

% weight function for inner products in Chebyshev basis
sq = chebfun(@(mu) 2/(T2(2)-T2(1)) * sqrt(1-((2*mu-T2(1)-T2(2))/(T2(2)-T2(1)))^2), T2, 'vectorize','splitting','on');

% derivatives of Usch via Chebfun
for ii=1:p
	dUsch{ii} = diff(Usch{ii});
end

% setting up the example, only one in the pamm-paper
switch example
  case{5}

		% setting up A(mu)
		for ii = 1:n2
			for jj = 1:n2
				M{ii,jj} = null;
				M{n2+ii,jj} = null;
				M{ii,n2+jj} = null;
				M{n2+ii,n2+jj} = null;
			end
		end 
		
		for ii = 1:n2
			% K stiffness matrix (2,1) block = -K
			if (ii<n2)
				M{n2+ii,ii+1} = eins;
			end
			M{n2+ii,ii} = -zwei;             
			if (ii>1)
				M{n2+ii,ii-1} = eins;
			end
			% N = I linearization (1,2)-block = I
			M{ii,n2+ii} = eins;             
			
			% damping matrix (2,2)-block = -D
			M{n2+ii,n2+ii} = chebfun(@(t) -5-5/(1+exp(-(n2-ii-t))), T2, 'vectorize','splitting','on');
		end

		% Setting up system matrix for trajectory
		Test = zeros(n,n);
		for ii = 1:n
			for jj = 1:n
				Test(ii,jj) = (h*M{ii,jj}(mu_hidden)); 
			end
		end
		
		Test = expm(Test);
		
end

% generate a tractory based on the hidden mu
x = zeros(n,realizations*timesteps);

for kk = 1:realizations
	if (isclose)
		x(:,(kk-1)*timesteps+1) = noise_eps*[randn(n2,1);randn(n2,1)];
	else
		x(:,(kk-1)*timesteps+1) = 10*randn(n,1);
	end 

	
	for ii = 2:timesteps
		%x(:,(kk-1)*timesteps+ii) = Test*x(:,(kk-1)*timesteps+ii-1) + noise_eps*[zeros(n2,1);randn(n2,1)];
		x(:,(kk-1)*timesteps+ii) = Test*x(:,(kk-1)*timesteps+ii-1) + noise_eps*[randn(n2,1);randn(n2,1)];
	end
end


% max degree 
if (~exist('p','var'))
	p = 25;
end

% max degree A 
if (~exist('pA','var'))
	pA = p;
end

% Taylor expansion point
if (~exist('t0','var'))
	t0 = 0.2;
end

% discretization of the parameter interval
xl = linspace(T(1),T(2),npoints);



tic

if (taylor) && (chebyshev)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% Taylor approximation

	% not implemented
	
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% Chebyshev approximation

	% generate Chebyshev polynomials of second kind, scaled
	gen_tsch;                     
	sq = chebfun(@(mu) 2/(T2(2)-T2(1)) * sqrt(1-((2*mu-T2(1)-T2(2))/(T2(2)-T2(1)))^2), T2, 'vectorize','splitting','on');

	Ac = zeros(n,n,p);            % Chebyshev expansion of M
	for ii = 1:n
		for jj = 1:n
			for kk = 1:min(p,pA)
				Ac(ii,jj,kk) = sum(M{ii,jj}*Usch{kk}*sq,T2(1),T2(2));
			end		
		end
	end	
	
elseif (taylor)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% Taylor approximation

	% not implemented
else
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% Chebyshev approximation
	
	% generate Chebyshev polynomials of second kind, scaled
	gen_tsch;                     
	sq = chebfun(@(mu) 2/(T2(2)-T2(1)) * sqrt(1-((2*mu-T2(1)-T2(2))/(T2(2)-T2(1)))^2), T2, 'vectorize','splitting','on');

	A = zeros(n,n,p);            % Chebyshev expansion of M
	for ii = 1:n
		for jj = 1:n
			for kk = 1:min(p,pA)
				A(ii,jj,kk) = sum(M{ii,jj}*Usch{kk}*sq,T2(1),T2(2));
			end		
		end
	end	
	
end

time_generating_or_computing_A = toc


tic
if (taylor && chebyshev)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% Taylor and Chebyshev approximation
	
	[dp,vp] = taylor_evp(A,p,pA,usesingle);
	[dpc,vpc] = cheb_evp(Ac,p,newtonsteps,pA,usesingle);
	
elseif (taylor)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% Taylor approximation

	[dp,vp] = taylor_evp(A,p,pA,usesingle);
	
else
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% Chebyshev approximation
	
	[dp,vp] = cheb_evp(A,p,newtonsteps,pA,usesingle);
	
end

time_solving_pevp = toc


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% float 1 Figure 2.1 sampled eigenvalues and blue lines
if (ismember(1,floats))
	
	Ax = zeros(n,n);
	makepgfplots_figure1;

end
	
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% float 2 Table 2.2 Coefficients Taylor approximations and their sum
if (ismember(2,floats))

	make_table2_2;

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% float 3 Figure 2.2/2.3 maximal difference between Taylor/Chebyshev and eigenvalues 
if (ismember(3,floats))
	
	makepgfplots_figure2;

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% float 4 Figure 2.2/2.3 maximal difference between Taylor/Chebyshev and eigenvectors 
if (ismember(4,floats))
	
	makepgfplots_figure3;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% float 5 Figure 2.2/2.3 maximal difference between eigenvalues and eigenvalues
% computed from the eigenvectors 
if (ismember(5,floats))
	
	makepgfplots_figure4;

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% float 6 Figure 4.5/4.6 sampled eigenvalues and blue lines real and complex parts
if (ismember(6,floats))
	
	makepgfplots_figure5;

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% float 6 Figure 6.1 both sampled eigenvalues and blue lines real and complex parts
if (ismember(7,floats))
	
	makepgfplots_figure6;
	
end






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% float 11 ksdensity
if (ismember(11,floats))
	
	makepgfplots_figure11;
	
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% float 21 ksdensity
if (ismember(21,floats))
	
	makepgfplots_figure21;
	
end









%%% Local Variables: 
%%% mode:matlab
%%% flyspell-mode:nil
%%% mode:flyspell-prog
