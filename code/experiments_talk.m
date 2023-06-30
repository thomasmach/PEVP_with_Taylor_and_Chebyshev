%% numerical experiments 
%% generating a test matrix

% we need chebfun
% add the correct folder if needed
addpath('~/git/chebfun')


% type of experiments, the ones for the talk
type = 'tal';

% selects different sets of parameters for different  numerical experiments
if (~exist('select','var'))
	select = 1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Table Select
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% select   floats            filenames      method                   example
% 1        slide 7 & 10        fig41          Taylor                   1 ... 2.3  
%                              tab41          Taylor                   1
% 2        slide 11            fig52          Taylor                   1 
% 3        slide 11            fig53          Taylor                   1
% 4          Fig 3.1           fig52_cheb     Chebyshev                1
% **** Fig 4.1 (sketch of the springs and masses) not a Matlab figure  1
% 5        slide 15            fig61          Taylor                   2 ... 4.1
% 6        slide 16            fig62          Taylor                   2   
% 7          Fig 4.2 (half)    fig61_cheb     Chebyshev                2
% 57         Fig 4.2 (both)    fig61_both     Taylor and Chebyshev     2
% 8          Fig 4.4           fig62_cheb     Chebyshev                2  
% 9          Fig 4.5           fig71          Taylor                   3 ... 4.2 
% 10         -------not used in paper-----    Taylor                   3
% 11         Fig 4.6           fig71_cheb     Chebyshev                3
% 12         Fig 4.7           fig61_taylor_1 Taylor                   1 ... 2.3
% 13         Fig 4.9           fig61_cheb_1   Chebyshev                1 
% 14         Fig 4.8           fig61_taylor_2 Taylor                   2 ... 4.1
% 15         Fig 4.10          fig61_cheb_2   Chebyshev                2
% 16         Fig 4.11          fig71_cheb_1   Chebyshev                1 ... 2.3
% 17         -------not used in paper-----    Taylor                   1 
% 18         -------not used in paper-----    Taylor                   1
% 101        -------not used in paper-----    Taylor                   4 ... ???
% for Fig 4.12, 4.13, 4.14 use experiments_sampling_2.m
% 41         -------not used in paper-----    Chebyshev                1
% 1001     slide 24            fig81          Chebyshev                1
% 2001     slide 24            fig81          Chebyshev                1

usevpa = false;
chebyshev = false;
fnns = false;
taylor = false;

% parameter interval (Chebyshev approximation)
T2 = [1/4,1];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% floats 
% 1  figures eigenvalues smapled and Taylor approximated, magnification
% 2  table sum
% 3  figures difference eigenvalues and approximation
% 4  like 3 but for Chebyshev
% 5  like 1 but example 2
% 6  like 3 but example 3
% 7  like 5 but Chebyshev
% 8  like 6 but Chebyshev
% 9  like 1 but example 3 mu = -0.2
% 10 like 1 but example 3 mu = 0.2

switch select
	case {1}
		% proof-of-concept Taylor and Table sum to 8
		floats = [1,2];

		n = 8;
		p = 7;
		% parameter interval (plotting)
		T = [0, 0.25];
		% parameter figure magnifying
		S = [0.01,0.07,0.8,2.1];

		usevpa = false;
		usesingle = false;

		taylor = true;
		
		example = 1;

		% parameter expansion point (Taylor approximation)
		t0 = 0.025;		
		
		% quality approximation with increasing degree
	case {2}
		floats = [3];

		n = 8;
		p = 26;
		% parameter interval
		T = [0, 0.25];		
		% parameter figures
		S = [0.00,0.05,1e-17,1e-11];
		usevpa = false;
		usesingle = false;

		taylor = true;
		
		example = 1;
		% parameter expansion point (Taylor approximation)
		t0 = 0.025;		

		% quality approximation with increasing degree --- single precision
	case {3}
		floats = [3];

		n = 8;
		p = 23;
		% parameter interval
		T = [0, 0.25];		
		% parameter figures
		S = [0.0,0.05,1e-10,1e-4];
		usevpa = false;
		usesingle = true;
		
		taylor = true;
		
		example = 1;
		% parameter expansion point (Taylor approximation)
		t0 = 0.025;		

		% quality approximation with increasing degree Chebyshev
	case {4}
		floats = [3];

		n = 8;
		p = 20;
		% parameter interval
		T = [0, 1.5];		
		% parameter figures
		S = [0.25,1.0,1e-16,1e-12];
		% parameter interval (Chebyshev approximation)
		T2 = [1/4,1];
		t0 = 0.5;
		usevpa = false;
		usesingle = false;
		
		taylor = false;
		
		example = 1;

	case {41}
		floats = [1];

		n = 8;
		p = 7;
		% parameter interval
		T = [0, 1.5];		
		% parameter figures
		S = [0.25,1.0,1e-16,1e-12];
		% parameter interval (Chebyshev approximation)
		T2 = [1/4,1];
		t0 = 0.5;
		usevpa = false;
		usesingle = false;
		
		taylor = false;
		
		example = 1;

		
	case {5}
 		floats = [1];
 
 		n = 8;
 		p = 7;
 		% parameter interval
 		T = [0.1, 1.6];
 		% parameter figures
 		S = [0.9,1.2,3.0,4.0];
 		y_lim = [0, 2];
 		usevpa = false;
 		usesingle = false;
 		
 		taylor = true;
 		
 		example = 2;
 		t0 = 0.8
	
	case {57} %plot both 5 and 7
 		floats = [7];
 
 		n = 8;
 		p = 7;
 		% parameter interval
 		T = [0.1, 1.6];
 		% parameter figures
 		S = [0.9,1.2,3.0,4.0];
 		y_lim = [0, 2];
 		usevpa = false;
 		usesingle = false;
 		
 		taylor = true;
		chebyshev = true;
 		
 		example = 2;
		T2 = [0.5,1];
 		t0 = 0.8
 		
 	case {6}
 		floats = [3];
 
 		n = 8;
		p = 26;
		% parameter interval
		T = [0.1, 1.6];		
		% parameter figures
		S = [0.5,1.0,1e-15,1e-11];
 		y_lim = [0, 2];
 		usevpa = false;
 		usesingle = false;
 
		taylor = true;
 		
 		example = 2;
 		t0 = 0.8;
 
	case {7}
 		floats = [1];
 
 		n = 8;
 		p = 7;
 		% parameter interval
 		T = [0.1, 1.6];
 		% parameter figures
 		S = [0.9,1.2,3.0,4.0];
 		y_lim = [0, 2];
 		usevpa = false;
 		usesingle = false;
 		
 		taylor = false;
 		
 		example = 2;
		T2 = [0.5,1];
 		t0 = 0.8
	
	case {8}
 		floats = [3];
 
 		n = 8;
		p = 30;
		% parameter interval
		T = [0.1, 1.6];		
		% parameter figures
		S = [0.5,1.0,1e-15,1e-11];
 		y_lim = [0, 2];
 		usevpa = false;
 		usesingle = false;
 
		taylor = false;
 		
 		example = 2;
		T2 = [0.5,1];
 		t0 = 0.8;
 		
		newtonsteps = 30; 
		
	case {9}
		% example 3
		floats = [6];

		n = 8;
		p = 26;
		% parameter interval (plotting)
		T = [-1.0, 1.0];
		% parameter figure magnifying
		S = [42,42,42,42];

		usevpa = false;
		usesingle = false;

		taylor = true;
		
		example = 3;

		% parameter expansion point (Taylor approximation)
		t0 = 0.2;		
		
	case {10}
		% example 3
		floats = [6];

		n = 8;
		p = 26;
		% parameter interval (plotting)
		T = [-1.0, 1.0];
		% parameter figure magnifying
		S = [42,42,42,42];

		usevpa = false;
		usesingle = false;

		taylor = true;
		
		example = 3;

		% parameter expansion point (Taylor approximation)
		t0 = -0.2;		

	case {11}
		% example 3
		floats = [6];

		n = 8;
		p = 20;
		% parameter interval (plotting)
		T = [-1.0, 1.0];
		% parameter figure magnifying
		S = [42,42,42,42];

		usevpa = false;
		usesingle = false;

		taylor = false;
		
		example = 3;

		T2 = [0.1,0.5];
		% parameter expansion point (Taylor approximation)
		t0 = -0.2;		


		%% float 4
	case {12}
		floats = [4];

		n = 8;
		p = 26;
		% parameter interval
		T = [0, 1.5];		
		% parameter figures
		S = [0.01,0.4,1e-16,1e-12];
		usevpa = false;
		usesingle = false;

		taylor = true;
		
		example = 1;
		% parameter expansion point (Taylor approximation)
		t0 = 0.2;		
		
	case {13}
		floats = [4];

		n = 8;
		p = 20;
		% parameter interval
		T = [0, 1.5];		
		% parameter figures
		S = [0.25,1.0,1e-16,1e-12];
		% parameter interval (Chebyshev approximation)
		T2 = [1/4,1];
		t0 = 0.5;
		usevpa = false;
		usesingle = false;
		
		taylor = false;
		
		example = 1;
		
 	case {14}
 		floats = [4];
 
 		n = 8;
		p = 26;
		% parameter interval
		T = [0.1, 1.6];		
		% parameter figures
		S = [0.5,1.0,1e-16,1e-12];
 		y_lim = [0, 2];
 		usevpa = false;
 		usesingle = false;
 
		taylor = true;
 		
 		example = 2;
 		t0 = 0.8;
		
	case {15}
 		floats = [4];
 
 		n = 8;
		p = 30;
		% parameter interval
		T = [0.1, 1.6];		
		% parameter figures
		S = [0.5,1.0,1e-17,1e-13];
 		y_lim = [0, 2];
 		usevpa = false;
 		usesingle = false;
 
		taylor = false;
 		
 		example = 2;
		T2 = [0.5,1];
 		t0 = 0.8;
 		
		newtonsteps = 30; 

 	case {16}
 		floats = [5];
 
		n = 8;
		p = 20;
		% parameter interval
		T = [0, 1.5];		
		% parameter figures
		S = [0.25,1.0,1e-16,1e-12];
		% parameter interval (Chebyshev approximation)
		T2 = [1/4,1];
		t0 = 0.5;
		usevpa = false;
		usesingle = false;
		
		taylor = false;
		
		example = 1;

	case {17}
 		floats = [5];

		n = 8;
		p = 26;
		% parameter interval
		T = [0, 1.5];		
		% parameter figures
		S = [0.01,0.4,1e-16,1e-12];
		usevpa = false;
		usesingle = false;

		taylor = true;
		
		example = 1;
		% parameter expansion point (Taylor approximation)
		t0 = 0.2;		

	case {18}
 		floats = [5];

		n = 800;
		p = 26;
		% parameter interval
		T = [0, 1.5];		
		% parameter figures
		S = [0.01,0.4,1e-16,1e-12];
		usevpa = false;
		usesingle = false;

		taylor = true;
		
		example = 1;
		% parameter expansion point (Taylor approximation)
		t0 = 0.2;		


	
	case {101}
		% example 4 based on paper by Leon Bungert and Philipp Wacker   
		
		floats = [3];

		n = 10;
		p = 20;
		% parameter interval
		T = [0, 2.5];		
		% parameter figures
		S = [0.0,1.0,1e-16,1e-12];
		% parameter interval (Chebyshev approximation)
		T2 = [0.0, 1.0];
		t0 = 1.0;
		usevpa = false;
		usesingle = false;
		
		taylor = false;
		
		example = 4;

		% parameters for example 4
		baart_s = 1e-3; % covariance matrix in Toeplitz form
		baart_r = 0.8; 
		

	case {1000,1001,1002,1003,1004,1005,1006,1007,1008,1009}
		floats = [3];

		n = 8;
		p = 25;
		% parameter interval
		T = [0, 0.25];		
		% parameter figures
		S = [42,42,42,42];
		usevpa = false;
		usesingle = false;

		taylor = false;
		chebyshev = true;
		
		example = 1;
		% parameter expansion point (Taylor approximation)
		T2 = [0.01,0.1];
		t0 = 0.05;

		fnns = true;
		newtonsteps = select - 1000; 
		
	case {2000,2001,2002,2003,2004,2005,2006,2007,2008,2009}
		floats = [3];

		n = 8;
		p = 40;
		% parameter interval
		T = [0, 0.25];		
		% parameter figures
		S = [42,42,42,42];
		usevpa = false;
		usesingle = false;

		taylor = false;
		chebyshev = true;
		
		example = 1;
		% parameter expansion point (Taylor approximation)
		T2 = [0.01,0.1];
		t0 = 0.05;

		fnns = true;
		newtonsteps = select - 2000; 
		
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if (~exist('output','var'))
	output = true;
end

npoints = 151;

% points
r1 = 5; % big radius torus
r2 = 1; % small radius torus

% windings
if (~exist('k','var'))
	k = 2; 
end

% size of points, dimension of A
if (~exist('n','var'))
	n = 8;
end

if (~exist('newtonsteps','var'))
	newtonsteps = 20;
end


null = chebfun(@(t) 0, T2, 'vectorize','splitting','on');
eins = chebfun(@(t) 1, T2, 'vectorize','splitting','on');


switch example
	case{1}
	rn = linspace(0,1,n+1);
	rn = rn(1:n);

	% points on the windings 
	if (~exist('P','var'))
		P = zeros(n,3);
		P(:,1) = cos(2*pi*rn).*(r1+r2*cos(2*pi*k*rn));
		P(:,2) = sin(2*pi*rn).*(r1+r2*cos(2*pi*k*rn));
		P(:,3) = r2*sin(2*pi*k*rn);
		n_to_make_P = n;
	else
		if (n~=n_to_make_P)
			P = zeros(n,3);
			P(:,1) = cos(2*pi*rn).*(r1+r2*cos(2*pi*k*rn));
			P(:,2) = sin(2*pi*rn).*(r1+r2*cos(2*pi*k*rn));
			P(:,3) = r2*sin(2*pi*k*rn);
			n_to_make_P = n;
		end
	end

	% setup points
	U = zeros(n,n);
	for ii = 1:n
		for jj = 1:n
			% sqrt is a different kernel than the standard Gaussian kernel
			% U(ii,jj) = sqrt((P(ii,:)-P(jj,:))*transpose(P(ii,:)-P(jj,:)));
			U(ii,jj) = (P(ii,:)-P(jj,:))*transpose(P(ii,:)-P(jj,:));
			h = chebfun(@(mu) exp(-(mu)*U(ii,jj)), T2, 'vectorize','splitting','on');
			% M = exp(-\mu U) \neq expm(U)
			M{ii,jj} = h;             
		end
	end

	case{2}
	% setup stiffness matrix
	K = diag(2*ones(n,1),0)-diag(ones(n-1,1),1)-diag(ones(n-1,1),-1);

	% functions for Chebyshev approximation
	for ii = 1:n
		if ((ii==floor(n/2)+1)||(ii==floor(n/2)+2))
			for jj=1:ii-2
				M{ii,jj}=null;
			end
			for jj=max(ii-1,1):min(ii+1,n)
				h = chebfun(@(mu) K(ii,jj)/mu, T2, 'vectorize','splitting','on');
				M{ii,jj} = h;             
			end
			for jj=ii+2:n
				M{ii,jj} = null;
			end
		else		
			for jj=1:ii-2
				M{ii,jj}=null;
			end
			for jj=max(ii-1,1):min(ii+1,n)
				h = chebfun(@(mu) K(ii,jj), T2, 'vectorize','splitting','on');
				M{ii,jj} = h;             
			end
			for jj=ii+2:n
				M{ii,jj} = null;
			end
		end
	end
	
	case{3}
		% functions for Chebyshev approximation
		for ii = 1:n-1
			for jj=1:ii-1
				M{ii,jj} = null;
			end
			for jj=ii:min(ii+1,n)
				M{ii,jj} = eins;
			end
			for jj=ii+2:n
				M{ii,jj} = null;
			end
		end		
		h = chebfun(@(mu) mu, T2, 'vectorize','splitting','on');
		M{n,1} = h;
		for jj=2:n-1
			M{n,jj} = null;
		end
		M{n,n} = eins;
		
	
  case{4}
		[baart_A,baart_b,baart_x] = baart(n);
		baart_fr = zeros(n,1);
		baart_fr(1) = [baart_s];
		for ii = 2:n
			baart_fr(ii) = baart_fr(ii-1)*baart_r;
		end
		baart_C = toeplitz(baart_fr)
	
		
end



% max degree 
if (~exist('p','var'))
	p = 25;
end

% max degree A 
if (~exist('pA','var'))
	pA = p;
end

% sampling
if (~exist('sampling','var'))
	sampling = 10000;
end

% drawrandommu
if (~exist('drawrandommu','var'))
	drawrandommu = @() randn()/30+0.2; % normal distributed around 0.2 with std deviation 1/30
	sampling = 10000;
end

% le ... eigenvalues to be observed
if (~exist('le','var'))
	le = [2,3];
end

% lebonus ... bonus eigenvalues
if (~exist('lebonus','var'))
	lebonus = 3;
end


% Taylor expansion point
if (~exist('t0','var'))
	t0 = 0.2;
end

% draw sampling points
store_mu_rand = zeros(sampling,1);
for ii = 1:sampling
	store_mu_rand(ii) = drawrandommu();
end

% discretization of the parameter interval
xl = linspace(T(1),T(2),npoints);






tic

if (taylor) && (chebyshev)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% Taylor approximation

	switch (example)
		case {1}
			A = zeros(n,n,p);
			for kk = 1:p
				%A(:,:,kk) = (-U).^(kk-1).*exp(-t0*U);
				A(:,:,kk) = (-U).^(kk-1).*exp(-t0*U)/factorial(kk-1);
			end
			
			
		case {2}
			A = zeros(n,n,p);
			A(:,:,1) = diag([ones(floor(n/2),1);(1/t0)*ones(2,1);ones(n-floor(n/2)-2,1)])*K;
			for kk = 2:p
				%A(:,:,kk) = diag([zeros(floor(n/2),1);factorial(kk-1)*(-1)^(kk-1)*t0^(-kk)*ones(2,1);zeros(n-floor(n/2)-2,1)])*K;
				A(:,:,kk) = diag([zeros(floor(n/2),1);(-1)^(kk-1)*t0^(-kk)*ones(2,1);zeros(n-floor(n/2)-2,1)])*K;
			end		
			
		case {3}
			A = zeros(n,n,p);
			A(:,:,1) = diag(ones(n,1),0) + diag(ones(n-1,1),1);
			A(n,1,1) = t0;
			
			A(n,1,2) = 1;
		
	end
	
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

	switch (example)
		case {1}
			A = zeros(n,n,p);
			for kk = 1:p
				% the implementation differs from the paper in the sense that we divide
				% A^(k) by k! to avoid having to compute binomial coefficients and factorials 
				%A(:,:,kk) = (-U).^(kk-1).*exp(-t0*U);
				A(:,:,kk) = (-U).^(kk-1).*exp(-t0*U)/factorial(kk-1);
			end
			
			
		case {2}
			A = zeros(n,n,p);
			A(:,:,1) = diag([ones(floor(n/2),1);(1/t0)*ones(2,1);ones(n-floor(n/2)-2,1)])*K;
			for kk = 2:p
				%A(:,:,kk) = diag([zeros(floor(n/2),1);factorial(kk-1)*(-1)^(kk-1)*t0^(-kk)*ones(2,1);zeros(n-floor(n/2)-2,1)])*K;
				A(:,:,kk) = diag([zeros(floor(n/2),1);(-1)^(kk-1)*t0^(-kk)*ones(2,1);zeros(n-floor(n/2)-2,1)])*K;
			end		
			
		case {3}
			A = zeros(n,n,p);
			A(:,:,1) = diag(ones(n,1),0) + diag(ones(n-1,1),1);
			A(n,1,1) = t0;
			
			A(n,1,2) = 1;
		
	end
	
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
	fprintf('here\n')

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





if (select == 2)
	% data for paragraph on page 9
	
	p = 5
	h = sum(dp(:,p+1))
	p = 10
	h = sum(dp(:,p+1))
	p = 15
	h = sum(dp(:,p+1))

end





















%%% Local Variables: 
%%% mode:matlab
%%% flyspell-mode:nil
%%% mode:flyspell-prog
