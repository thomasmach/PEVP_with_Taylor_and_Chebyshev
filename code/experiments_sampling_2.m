%% numerical experiments 
%% generating a test matrix

% Example 1: kernel method exp(-mu U)
% Example 2: parametric PDE
% Example 3: Jordan block
% Example 4: symmetric version of the parametric PDE

% we need chebfun
% add the correct folder if needed
addpath('~/git/chebfun')


% selects different sets of parameters for different  numerical experiments
if (~exist('select','var'))
	select = 1;
end
if (~exist('output','var'))
	output = false;
end
usevpa = false;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Table select
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% select   floats            filenames             example
% 1          Fig 4.12/13       dat1                  1 ... 2.3  
% 2          Fig 4.14          dat2                  2 ... 4.1  
% 3         -------not used in paper-----
% 4         -------not used in paper-----
% 5         -------not used in paper-----
% 6         -------not used in paper-----
% 7         -------not used in paper-----


switch select
	case {1}
		n = 8;
		p = 7;

		% parameter interval (plotting)
		T = [0, 1.5];

		usevpa = false;
		usesingle = false;

		
		example = 1;
		% parameter interval (Chebyshev approximation)
		T2 = [0.1,0.3];
		t0 = 0.2;

		
		% random generator for sampling points
		drawrandommu = @() randn()/10+0.2; % normal distributed around 0.2 with std deviation 1/5

		% no of points
		sampling = 10000;
		
		% which eigenvalues
		le = [2 3];
		len = n;

	case {2}
		n = 8;
		p = 7;

		% parameter interval (plotting)
		T = [0.1, 1.6];

		usevpa = false;
		usesingle = false;

		
		example = 2;
		% parameter interval (Chebyshev approximation)
		T2 = [0.6,1.0];
		t0 = 0.8;

		
		% random generator for sampling points
		drawrandommu = @() randn()/10+0.8; % normal distributed around 0.2 with std deviation 1/5

		% no of points
		sampling = 10000;
		
		% which eigenvalues
		le = [2 3];
		len = n;

	case {3}
		n = 8;
		p = 7;

		% parameter interval (plotting)
		T = [0.1, 1.6];

		usevpa = false;
		usesingle = false;

		
		example = 4;
		% parameter interval (Chebyshev approximation)
		T2 = [0.6,1.0];
		t0 = 0.8;

		
		% random generator for sampling points
		drawrandommu = @() randn()/10+0.8; % normal distributed around 0.2 with std deviation 1/5

		% no of points
		sampling = 10000;
		
		% which eigenvalues
		le = [2 3];
		len = n;

	case {4}
		n = 8;
		p = 7;

		% parameter interval (plotting)
		T = [0.1, 1.6];

		usevpa = false;
		usesingle = false;

		
		example = 5;
		% parameter interval (Chebyshev approximation)
		T2 = [0.6,1.0];
		t0 = 0.8;

		
		% random generator for sampling points
		drawrandommu = @() randn()/10+0.8; % normal distributed around 0.2 with std deviation 1/5

		% no of points
		sampling = 10000;
		
		% which eigenvalues
		le = [2 3];
		len = n;

		
	case {5}
		n = 80;
		p = 7;

		% parameter interval (plotting)
		T = [0, 1.5];

		usevpa = false;
		usesingle = false;

		
		example = 1;
		% parameter interval (Chebyshev approximation)
		T2 = [0.1,0.3];
		t0 = 0.2;

		
		% random generator for sampling points
		drawrandommu = @() randn()/10+0.2; % normal distributed around 0.2 with std deviation 1/5

		% no of points
		sampling = 10000;
		
		% which eigenvalues
		le = [2 3];
		len = n;
	
	case {6}
		n = 800;
		p = 7;

		% parameter interval (plotting)
		T = [0, 1.5];

		usevpa = false;
		usesingle = false;

		
		example = 1;
		% parameter interval (Chebyshev approximation)
		T2 = [0.1,0.3];
		t0 = 0.2;

		
		% random generator for sampling points
		drawrandommu = @() randn()/10+0.2; % normal distributed around 0.2 with std deviation 1/5

		% no of points
		sampling = 10000;
		
		% which eigenvalues
		le = [2 3];
		len = n;
	
	case {7}
		n = 8000;
		p = 7;

		% parameter interval (plotting)
		T = [0, 1.5];

		usevpa = false;
		usesingle = false;

		
		example = 1;
		% parameter interval (Chebyshev approximation)
		T2 = [0.1,0.3];
		t0 = 0.2;

		
		% random generator for sampling points
		drawrandommu = @() randn()/10+0.2; % normal distributed around 0.2 with std deviation 1/5

		% no of points
		sampling = 10000;
		
		% which eigenvalues
		le = [2 3];
		len = n;
		
end


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

% le ... eigenvalues to be observed
if (~exist('le','var'))
	le = [2,3];
end

null = chebfun(@(t) 0, T2, 'vectorize','splitting','on');
eins = chebfun(@(t) 1, T2, 'vectorize','splitting','on');

if (example == 1)
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
			U(ii,jj) = sqrt((P(ii,:)-P(jj,:))*transpose(P(ii,:)-P(jj,:)));
			h = chebfun(@(mu) exp(-(mu)*U(ii,jj)), T2, 'vectorize','splitting','on');
			% M = exp(-\mu U) \neq expm(U)
			M{ii,jj} = h;             
		end
	end
end

if (example == 2 || example == 5)
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
	
end

if (example == 3)
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
	
end

if (example == 4)
	% setup stiffness matrix
	K = diag(2*ones(n,1),0)-diag(ones(n-1,1),1)-diag(ones(n-1,1),-1);

	% functions for Chebyshev approximation
	for ii=1:n
		for jj=1:n	
			if (abs(ii-jj)<=1)
				h = chebfun(@(mu) K(ii,jj), T2, 'vectorize','splitting','on');
				M{ii,jj} = h;             
			else				
				M{ii,jj}=null;
			end
		end
	end

	% floor(n/2), floor(n/2)+1, mu^-1/2
	M{floor(n/2),floor(n/2)+1} = chebfun(@(mu) -1/sqrt(mu), T2, 'vectorize','splitting','on');
	% floor(n/2)+1, floor(n/2), mu^-1/2
	M{floor(n/2)+1,floor(n/2)} = chebfun(@(mu) -1/sqrt(mu), T2, 'vectorize','splitting','on');
	% floor(n/2)+1, floor(n/2)+1, mu^-1
	M{floor(n/2)+1,floor(n/2)+1} = chebfun(@(mu) 2/mu, T2, 'vectorize','splitting','on');
	% floor(n/2)+1, floor(n/2)+2, mu^-1
	M{floor(n/2)+1,floor(n/2)+2} = chebfun(@(mu) -1/mu, T2, 'vectorize','splitting','on');
	% floor(n/2)+2, floor(n/2)+1, mu^-1
	M{floor(n/2)+2,floor(n/2)+1} = chebfun(@(mu) -1/mu, T2, 'vectorize','splitting','on');
	% floor(n/2)+2, floor(n/2)+2, mu^-1
	M{floor(n/2)+2,floor(n/2)+2} = chebfun(@(mu) 2/mu, T2, 'vectorize','splitting','on');
	% floor(n/2)+2, floor(n/2)+3, mu^-1/2
	M{floor(n/2)+2,floor(n/2)+3} = chebfun(@(mu) -1/sqrt(mu), T2, 'vectorize','splitting','on');
	% floor(n/2)+3, floor(n/2)+2, mu^-1/2
	M{floor(n/2)+3,floor(n/2)+2} = chebfun(@(mu) -1/sqrt(mu), T2, 'vectorize','splitting','on');
	
end


tic

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Taylor approximation

switch (example)
	case {1}
		AT = zeros(n,n,p);
		for kk = 1:p
			AT(:,:,kk) = (-U).^(kk-1).*exp(-t0*U);
		end
		
		
	case {2,5}
		AT = zeros(n,n,p);
		AT(:,:,1) = diag([ones(floor(n/2),1);(1/t0)*ones(2,1);ones(n-floor(n/2)-2,1)])*K;
		for kk = 2:p
			AT(:,:,kk) = diag([zeros(floor(n/2),1);factorial(kk-1)*(-1)^(kk-1)*t0^(-kk)*ones(2,1);zeros(n-floor(n/2)-2,1)])*K;
		end		
		
	case {3}
		AT = zeros(n,n,p);
		AT(:,:,1) = diag(ones(n,1),0) + diag(ones(n-1,1),1);
		AT(n,1,1) = t0;
			
		AT(n,1,2) = 1;
		
	case {4}
		AT = zeros(n,n,p);
		AT(:,:,1) = diag([ones(floor(n/2),1);(1/sqrt(t0))*ones(2,1);ones(n-floor(n/2)-2,1)])*K*diag([ones(floor(n/2),1);(1/sqrt(t0))*ones(2,1);ones(n-floor(n/2)-2,1)]);
		for kk = 2:p

			kl = kk;
			% setup stiffness matrix
			AT(:,:,kk) = zeros(n,n);
			% floor(n/2), floor(n/2)+1, mu^-1/2
			AT(floor(n/2),floor(n/2)+1,kk) = (-1)^mod(kl,2) * t0^(-(2*kl-1)/2) * prod(3:2:(2*kl-3))/2^(kl-1);
			% floor(n/2)+1, floor(n/2), mu^-1/2
			AT(floor(n/2)+1,floor(n/2),kk) = (-1)^mod(kl,2) * t0^(-(2*kl-1)/2) * prod(3:2:(2*kl-3))/2^(kl-1);
			% floor(n/2)+1, floor(n/2)+1, mu^-1
			AT(floor(n/2)+1,floor(n/2)+1,kk) = 2*factorial(kl-1)*(-1)^mod(kl+1,2) * t0^(-kl);
			% floor(n/2)+1, floor(n/2)+2
			AT(floor(n/2)+1,floor(n/2)+2,kk) = factorial(kl-1)*(-1)^mod(kl,2) * t0^(-kl);
			% floor(n/2)+2, floor(n/2)+1
			AT(floor(n/2)+2,floor(n/2)+1,kk) = factorial(kl-1)*(-1)^mod(kl,2) * t0^(-kl);
			% floor(n/2)+2, floor(n/2)+2
			AT(floor(n/2)+2,floor(n/2)+2,kk) = 2*factorial(kl-1)*(-1)^mod(kl+1,2) * t0^(-kl);
			% floor(n/2)+2, floor(n/2)+3, mu^-1/2
			AT(floor(n/2)+2,floor(n/2)+3,kk) = (-1)^mod(kl,2) * t0^(-(2*kl-1)/2) * prod(3:2:(2*kl-3))/2^(kl-1);
			% floor(n/2)+3, floor(n/2)+2, mu^-1/2
			AT(floor(n/2)+3,floor(n/2)+2,kk) = (-1)^mod(kl,2) * t0^(-(2*kl-1)/2) * prod(3:2:(2*kl-3))/2^(kl-1);

		end		
		
end
	
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Chebyshev approximation

% generate Chebyshev polynomials of second kind, scaled
gen_tsch;                     
sq = chebfun(@(mu) 2/(T2(2)-T2(1)) * sqrt(1-((2*mu-T2(1)-T2(2))/(T2(2)-T2(1)))^2), T2, 'vectorize','splitting','on');

AC = zeros(n,n,p);            % Chebyshev expansion of M
for ii = 1:n
	for jj = 1:n
		for kk = 1:min(p,p)
			AC(ii,jj,kk) = sum(M{ii,jj}*Usch{kk}*sq,T2(1),T2(2));
		end		
	end
end	

time_generating_or_computing_A = toc


tic
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Taylor approximation

[dpT,vpT] = taylor_evp(AT,p,p,usesingle);
	

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Chebyshev approximation

[dpC,vpC] = cheb_evp(AC,p,newtonsteps,p,usesingle);
	

time_solving_pevp = toc



tic
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% draw sampling points

store_mu_rand = zeros(sampling,1);
for ii = 1:sampling
	store_mu_rand(ii) = drawrandommu();
end

time_draw_sampling_points = toc



tic
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% full eigenvalue decomposition
for ii = 1:sampling
	murand = store_mu_rand(ii);

	switch (example)
		case {1}
			AA = exp(-murand*U);
		case {2,5}
			AA = diag([ones(floor(n/2),1);(1/murand)*ones(2,1);ones(n-floor(n/2)-2,1)])*K;
		case {4}
			AA = diag([ones(floor(n/2),1);(1/sqrt(murand))*ones(2,1);ones(n-floor(n/2)-2,1)])*K*diag([ones(floor(n/2),1);(1/sqrt(murand))*ones(2,1);ones(n-floor(n/2)-2,1)]);
	end
	
	% [ev,h] = eigs(AA,len,'largestreal');
	% h = diag(h);
	h = eigs(AA,len,'largestreal');
	store_sampled_eig_exact(:,ii) = h(le);
	
end
time_sampling_full_matrix = toc




tic 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sampling Taylor + Horner scheme
for ii = 1:sampling
	murand = store_mu_rand(ii);
	
	% h = fp{p}(murand);
	h = horner_f(murand,t0,dpT(:,1:p));
	h = sort(h,'descend');
	store_sampled_eig_taylor(:,ii) = h(le);
	
end
time_sampling_Taylor_Horner = toc


tic 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sampling Taylor + Horner scheme + Rayleigh
for ii = 1:sampling
	murand = store_mu_rand(ii);
	
	h5 = horner_f(murand,t0,vpT(:,:,1:p));
	switch (example)
		case {1}
			h6 = diag(h5'*exp(-murand*U)*h5)./diag(h5'*h5);
		case {2}
			h6 = diag(h5'*diag([ones(floor(n/2),1);(1/murand)*ones(2,1);ones(n-floor(n/2)-2,1)])*K*h5)./diag(h5'*h5);
		case {4}
			h6 = diag(h5'*diag([ones(floor(n/2),1);(1/sqrt(murand))*ones(2,1);ones(n-floor(n/2)-2,1)])*K*diag([ones(floor(n/2),1);(1/sqrt(murand))*ones(2,1);ones(n-floor(n/2)-2,1)])*h5)./diag(h5'*h5);
		case {5}
			h6 = diag(h5'*K*h5)./diag(h5'*diag([ones(floor(n/2),1);(murand)*ones(2,1);ones(n-floor(n/2)-2,1)])*h5);
	end			
	
	h6 = sort(h6,'descend');
	store_sampled_eig_taylor_rayleigh(:,ii) = h6(le);
	
end
time_sampling_Taylor_Horner_Rayleigh = toc


tic 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sampling Chebyshev
%fp{1} = @(mu) dpC(:,1)*Usch{1}(mu);
%
%for kk = 2:p
%	fp{kk} = @(mu) fp{kk-1}(mu) + dpC(:,kk)*Usch{kk}(mu);
%end

scfactor = sqrt(2)/sqrt(pi);
for ii = 1:sampling
	murand = store_mu_rand(ii);
	mutrans = 2*(murand-0.5*T2(1)-0.5*T2(2))/(T2(2)-T2(1));
	
	u1 = scfactor;
	h2 = dpC(:,1)*u1;
	if (p>=2)
 		u2 = 2*(mutrans)*scfactor;
		h2 = h2 + dpC(:,2)*u2;
	end
	for jj = 3:p
		u3 = 2*u2*mutrans-u1;
		u1 = u2;
		u2 = u3;
		h2 = h2 + dpC(:,jj)*u2;
	end
	
	h2 = sort(h2,'descend');
	store_sampled_eig_cheb(:,ii) = h2(le);
	
end
time_sampling_Chebyshev = toc


tic 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sampling Chebyshev with Rayleighquotient based on eigenvectors
for ii = 1:sampling
	murand = store_mu_rand(ii);
	mutrans = 2*(murand-0.5*T2(1)-0.5*T2(2))/(T2(2)-T2(1));
	
	u1 = scfactor;
	h3 = vpC(:,:,1)*u1;
	if (p>=2)
 		u2 = 2*(mutrans)*scfactor;
		h3 = h3 + vpC(:,:,2)*u2;
	end
	for jj = 3:p
		u3 = 2*u2*mutrans-u1;
		u1 = u2;
		u2 = u3;
		h3 = h3 + vpC(:,:,jj)*u2;
	end
	
	switch (example)
		case {1}
			h4 = diag(h3'*exp(-murand*U)*h3)./diag(h3'*h3);
		case {2}
			h4 = diag(h3'*diag([ones(floor(n/2),1);(1/murand)*ones(2,1);ones(n-floor(n/2)-2,1)])*K*h3)./diag(h3'*h3);
		case {4}
			h4 = diag(h3'*diag([ones(floor(n/2),1);(1/sqrt(murand))*ones(2,1);ones(n-floor(n/2)-2,1)])*K*diag([ones(floor(n/2),1);(1/sqrt(murand))*ones(2,1);ones(n-floor(n/2)-2,1)])*h3)./diag(h3'*h3);
		case {5}
			h4 = diag(h3'*K*h3)./diag(h3'*diag([ones(floor(n/2),1);(murand)*ones(2,1);ones(n-floor(n/2)-2,1)])*h3);
	end			

	h4 = sort(h4,'descend');
	store_sampled_eig_cheb_rayleigh(:,ii) = h4(le);
	
end
time_sampling_Chebyshev_Rayleigh = toc


tic 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sampling with v(\mu_0) and Rayleighquotient 
switch (example)
	case {1}
		[v02,d02] = eig(exp(-t0*U));
	case {2,5}
		[v02,d02] = eig(diag([ones(floor(n/2),1);(1/t0)*ones(2,1);ones(n-floor(n/2)-2,1)])*K);
	case {4}
		[v02,d02] = eig(diag([ones(floor(n/2),1);(1/sqrt(t0))*ones(2,1);ones(n-floor(n/2)-2,1)])*K*diag([ones(floor(n/2),1);(1/sqrt(t0))*ones(2,1);ones(n-floor(n/2)-2,1)]));
end

for ii = 1:sampling
	murand = store_mu_rand(ii);

	switch (example)
		case {1}
			h8 = diag(v02'*exp(-murand*U)*v02)./diag(v02'*v02);
		case {2}
			h8 = diag(v02'*diag([ones(floor(n/2),1);(1/murand)*ones(2,1);ones(n-floor(n/2)-2,1)])*K*v02)./diag(v02'*v02);
		case {4}
			h8 = diag(v02'*diag([ones(floor(n/2),1);(1/sqrt(murand))*ones(2,1);ones(n-floor(n/2)-2,1)])*K*diag([ones(floor(n/2),1);(1/sqrt(murand))*ones(2,1);ones(n-floor(n/2)-2,1)])*v02)./diag(v02'*v02);
		case {5}
			h8 = diag(v02'*K*v02)./diag(v02'*diag([ones(floor(n/2),1);(murand)*ones(2,1);ones(n-floor(n/2)-2,1)])*v02);
	end			
	h8 = sort(h8,'descend');
	store_sampled_eig_rayleigh(:,ii) = h8(le);
	
end
time_sampling_Rayleigh = toc


if (usevpa)
	filename = sprintf('exp_sampling_1_%d_%d_%f_%d_vpa.dat',n,p,t0,example);
else
	filename = sprintf('exp_sampling_1_%d_%d_%f_%d.dat',n,p,t0,example);
end

datout = fopen(filename,'w');

[~,ind]=sort(store_mu_rand);

fprintf(datout,'No mu exact1 exact2 rayleigh1 rayleigh2 taylor1 taylor2 taylor_ray1 taylor_ray2 cheb1 cheb2 cheb_ray1 cheb_ray2 err_rayleigh1 err_rayleigh2 err_taylor1 err_taylor2 err_taylor_ray1 err_taylor_ray2 err_cheb1 err_cheb2 err_cheb_ray1 err_cheb_ray2\n');
fprintf(datout,'%d %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e\n',[1:sampling;transpose(store_mu_rand(ind));store_sampled_eig_exact(:,ind);store_sampled_eig_rayleigh(:,ind);store_sampled_eig_taylor(:,ind);store_sampled_eig_taylor_rayleigh(:,ind);store_sampled_eig_cheb(:,ind);store_sampled_eig_cheb_rayleigh(:,ind);store_sampled_eig_exact(:,ind)-store_sampled_eig_rayleigh(:,ind);store_sampled_eig_exact(:,ind)-store_sampled_eig_taylor(:,ind);store_sampled_eig_exact(:,ind)-store_sampled_eig_taylor_rayleigh(:,ind);store_sampled_eig_exact(:,ind)-store_sampled_eig_cheb(:,ind);store_sampled_eig_exact(:,ind)-store_sampled_eig_cheb_rayleigh(:,ind)]);

fclose(datout);
	
if (output)
	h = figure;
	set(h,'WindowStyle','docked');
	plot(store_mu_rand(ind),store_sampled_eig_exact(1,ind),'r');
	hold on
	plot(store_mu_rand(ind),store_sampled_eig_taylor_rayleigh(1,ind),'k--');
	plot(store_mu_rand(ind),store_sampled_eig_cheb(1,ind),'g');
	plot(store_mu_rand(ind),store_sampled_eig_cheb_rayleigh(1,ind),'g--');
	plot(store_mu_rand(ind),store_sampled_eig_taylor(1,ind),'b');
	plot(store_mu_rand(ind),store_sampled_eig_taylor_rayleigh(1,ind),'b--');
	hold off
	
	h2 = figure;
	set(h2,'WindowStyle','docked');
	plot(store_mu_rand(ind),store_sampled_eig_exact(2,ind),'r');
	hold on
	plot(store_mu_rand(ind),store_sampled_eig_taylor_rayleigh(2,ind),'k--');
	plot(store_mu_rand(ind),store_sampled_eig_cheb(2,ind),'g');
	plot(store_mu_rand(ind),store_sampled_eig_cheb_rayleigh(2,ind),'g--');
	plot(store_mu_rand(ind),store_sampled_eig_taylor(2,ind),'b');
	plot(store_mu_rand(ind),store_sampled_eig_taylor_rayleigh(2,ind),'b--');
	hold off

	h3 = figure;
	set(h3,'WindowStyle','docked');
	semilogy(store_mu_rand(ind),abs(store_sampled_eig_taylor_rayleigh(2,ind)-store_sampled_eig_exact(2,ind)),'k--');
	hold on
	semilogy(store_mu_rand(ind),abs(store_sampled_eig_cheb(2,ind)-store_sampled_eig_exact(2,ind)),'g');
	semilogy(store_mu_rand(ind),abs(store_sampled_eig_cheb_rayleigh(2,ind)-store_sampled_eig_exact(2,ind)),'g--');
	semilogy(store_mu_rand(ind),abs(store_sampled_eig_taylor(2,ind)-store_sampled_eig_exact(2,ind)),'b');
	semilogy(store_mu_rand(ind),abs(store_sampled_eig_taylor_rayleigh(2,ind)-store_sampled_eig_exact(2,ind)),'b--');
	hold off

	h4 = figure;
	set(h4,'WindowStyle','docked');
	semilogy(store_mu_rand(ind),abs(store_sampled_eig_taylor_rayleigh(1,ind)-store_sampled_eig_exact(1,ind)),'k--');
	hold on
	semilogy(store_mu_rand(ind),abs(store_sampled_eig_cheb(1,ind)-store_sampled_eig_exact(1,ind)),'g');
	semilogy(store_mu_rand(ind),abs(store_sampled_eig_cheb_rayleigh(1,ind)-store_sampled_eig_exact(1,ind)),'g--');
	semilogy(store_mu_rand(ind),abs(store_sampled_eig_taylor(1,ind)-store_sampled_eig_exact(1,ind)),'b');
	semilogy(store_mu_rand(ind),abs(store_sampled_eig_taylor_rayleigh(1,ind)-store_sampled_eig_exact(1,ind)),'b--');
	hold off

end

%%% Local Variables: 
%%% mode:matlab
%%% flyspell-mode:nil
%%% mode:flyspell-prog
