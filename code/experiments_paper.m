%% numerical experiments 
%% generating a test matrix

% we need chebfun
% add the correct folder if needed
addpath('~/git/chebfun')


% selectors numerical experiments
if (~exist('selector','var'))
	selector = 1;
end
usevpa = false;

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
%
%

switch selector
	case {1}
		% proof-of-concept Taylor and Table sum to 8
		floats = [1,2];

		n = 8;
		md = 7;
		% parameter interval (plotting)
		T = [0, 1.5];
		% parameter figure magnifying
		S = [0.1,0.5,1,1.5];

		usevpa = false;
		usesingle = false;

		taylor = true;
		
		example = 1;

		% parameter expansion point (Taylor approximation)
		t0 = 0.2;		
		
		% quality approximation with increasing degree
	case {2}
		floats = [3];

		n = 8;
		md = 26;
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

		% quality approximation with increasing degree --- single precision
	case {3}
		floats = [3];

		n = 8;
		md = 18;
		% parameter interval
		T = [0, 1.5];		
		% parameter figures
		S = [0.01,0.4,1e-8,1e-4];
		usevpa = false;
		usesingle = true;
		
		taylor = true;
		
		example = 1;

		% quality approximation with increasing degree Chebyshev
	case {4}
		floats = [3];

		n = 8;
		md = 20;
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
 		md = 7;
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
 		
 	case {6}
 		floats = [3];
 
 		n = 8;
		md = 26;
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
 
	case {7}
 		floats = [1];
 
 		n = 8;
 		md = 7;
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
		md = 30;
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
		floats = [1];

		n = 8;
		md = 26;
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
		floats = [1];

		n = 8;
		md = 26;
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
		floats = [1];

		n = 8;
		md = 20;
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
		md = 26;
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
		md = 20;
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
		md = 26;
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
		md = 30;
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

 	case {16}
 		floats = [5];
 
		n = 8;
		md = 20;
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
		md = 26;
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
		md = 26;
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
		md = 20;
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
			U(ii,jj) = sqrt((P(ii,:)-P(jj,:))*transpose(P(ii,:)-P(jj,:)));
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
if (~exist('md','var'))
	md = 25;
end

% max degree A 
if (~exist('mdA','var'))
	mdA = md;
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

if (taylor)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% Taylor approximation

	switch (example)
		case {1}
			A = zeros(n,n,md);
			for kk = 1:md
				A(:,:,kk) = (-U).^(kk-1).*exp(-t0*U);
			end
			
			
		case {2}
			A = zeros(n,n,md);
			A(:,:,1) = diag([ones(floor(n/2),1);(1/t0)*ones(2,1);ones(n-floor(n/2)-2,1)])*K;
			for kk = 2:md
				A(:,:,kk) = diag([zeros(floor(n/2),1);factorial(kk-1)*(-1)^(kk-1)*t0^(-kk)*ones(2,1);zeros(n-floor(n/2)-2,1)])*K;
			end		
			
		case {3}
			A = zeros(n,n,md);
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

	A = zeros(n,n,md);            % Chebyshev expansion of M
	for ii = 1:n
		for jj = 1:n
			for kk = 1:min(md,mdA)
				A(ii,jj,kk) = sum(M{ii,jj}*Usch{kk}*sq,T2(1),T2(2));
			end		
		end
	end	
	
end

time_generating_or_computing_A = toc


tic
if (taylor)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% Taylor approximation

	[dp,vp] = taylor_evp(A,md,mdA,usesingle);
	
else
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% Chebyshev approximation
	
	[dp,vp] = cheb_evp(A,md,newtonsteps,mdA,usesingle);
	
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

























if (1==0)




for ii = 1:sampling
	murand = store_mu_rand(ii);
	
	% h = fp{md}(murand);
	hh = horner_f(murand,t0,dp(:,1:md));
	hh = sort(h,'descend');
	store_sampled_eig(:,ii) = hh(le);
	%store_sampled_eig2(:,ii) = max(0,min(h(le),n));
	
	
end
t = toc;



fprintf('n %d md %d time %8.4f sampling %d\n',n,md,t',sampling);


tic
for ii = 1:sampling
	murand = store_mu_rand(ii);
	
	AA = exp(-murand*U);
	[ev,h] = eigs(AA,max(le)+lebonus,'largestreal');
	h = diag(h);
	store_sampled_eig_exact(:,ii) = h(le);
	
end
toc

for ii = 1:sampling
	murand = store_mu_rand(ii);
	
	AA = exp(-murand*U);
	[ev,h] = eigs(AA,max(le)+lebonus,'largestreal');
	h = diag(h);
	store_sampled_eig_exact(:,ii) = h(le);	
	
	hpoly = horner_f(murand,t0,dp(:,1:md));

	[hpoly,ind] = sort(hpoly,'descend');
	%evpoly = fvp{md}(murand);
	evpoly = horner_f(murand,t0,vp(:,:,1:md));
	%norm(evpoly-evpoly2)

	evpoly = evpoly(:,ind);


	
	for kk = 1:length(le)
		error_eig_vec(kk,ii) = subspace(ev(:,le(kk)),evpoly(:,le(kk)));
	end

	for kk = 0:length(le)-1
		proj_ev(ii,(1:3)+3*kk) = PR'*ev(:,le(kk+1));
		proj_ev_exact(ii,(1:3)+3*kk) = PR'*evpoly(:,le(kk+1));
	end
	
	%if (max(error_eig_vec(:,ii))>0.1)
	%	%keyboard
	%	murand
	%end
	
	
end


if (output)
	if (~exist('h1','var'))
		h1 = figure;
		set(h1,'WindowStyle','docked');
	else
		figure(h1);
		clf
	end
	hold on

	%fp{1} = @(mu) dp(:,1)+(mu-mu);
	%fvp{1} = @(mu) vp(:,:,1)+(mu-mu);

	%for kk = 2:md
	%	fp{kk} = @(mu) fp{kk-1}(mu) + dp(:,kk)/factorial(kk-1)*(mu-t0).^(kk-1);
	%	fvp{kk} = @(mu) fvp{kk-1}(mu) + vp(:,:,kk)/factorial(kk-1)*(mu-t0).^(kk-1);
	%end

%	plot(xl,fp{md-1}(xl),'b--');
	plot(xl,horner_f(xl,t0,dp(:,1:md-1)),'b--');
%	plot(xl,fp{md}(xl),'b');
	plot(xl,horner_f(xl,t0,dp(:,1:md)),'b--');
	ylim(y_lim);
	
	legend('pu_p','u_p','Location','northeastoutside');

	filename = sprintf('sevp7_example1_%d_%d_%f_%d.tex',n,md,t0,example);
	figout = fopen(filename,'w');
	
	fprintf(figout,'\\begin{figure}\n');
	fprintf(figout,'\\centering\n');
	fprintf(figout,'\\beginpgfgraphicnamed{figure2-%d-%d-%d}\n',n,md,round(t0*100));
	fprintf(figout,'\\begin{tikzpicture}\n');
	fprintf(figout,'  \\begin{semilogyaxis}[ %%\n');
	fprintf(figout,'    remember picture, %%\n');
	fprintf(figout,'    name = plot4a, %%\n');
	fprintf(figout,'    scale only axis, %%\n');
	fprintf(figout,'    scaled y ticks = false,%%\n');
	fprintf(figout,'    width=0.75\\textwidth,%%\n');
	fprintf(figout,'    height=0.22067\\textheight,%%\n');
	fprintf(figout,'    xmin = %e, xmax = %e, %% change if needed\n', T(1), T(2));
	fprintf(figout,'    ymin = %e, ymax = %e, %%\n', 1e-18, 1e6);
	fprintf(figout,'    ytick={1e-16,1e-12,1e-8,1e-4,1e0,1e4}, %%\n');
	fprintf(figout,'    xlabel = {$\\mu$},%%\n');
	fprintf(figout,'    ylabel = {Error},%%\n');
	fprintf(figout,'    scaled ticks = false,%%\n');
	fprintf(figout,'    every axis y label/.style= {at={(ticklabel cs:0.5)},rotate=90,yshift=3pt},%%\n');
	fprintf(figout,'    axis on top, %%\n');
	fprintf(figout,'    every axis legend/.append style={ at={(0.00,1.05)}, anchor=south west,\n');
	fprintf(figout,'      cells={anchor=west}, legend columns=2},%%\n');
	fprintf(figout,'    every axis post/.style={thick, no marks},%%\n');
	fprintf(figout,'    mark size = 2]\n\n');
 

	z = zeros(npoints,md);
	for kk = 1:npoints
		xx = xl(kk);
		for ii = 1:n
			for jj = 1:n
				Ax(ii,jj) = exp(-xx*U(ii,jj));
			end
		end
		[vv,e] = eig(Ax);
		[e,ind] = sort(diag(e),'descend');
		plot(xx*ones(n,1),e,'r+');
		for ll = 1:md
			[e2,ind2] = sort(horner_f(xx,t0,dp(:,1:ll)),'descend');
			z(kk,ll) = norm(sort(e(1:len),'descend')-e2);
			zv(kk,ll) = subspace(vv(:,ind(2)),horner_f(xx,t0,vp(:,ind2(2),1:ll)));
			%z(kk,ll) = norm(sort(e(1:len))-sort(fp{ll}(xx)));
		end
	end

	
	
	for ii=1:md
		fprintf(figout,'    \\addplot\n');
		fprintf(figout,'    coordinates{%%\n');
		el = zeros(1,n); 
		el(ii) = 1;
		for kk = 1:npoints
			fprintf(figout,'    (%e,%e)%%\n',xl(kk),z(kk,ii));
		end			
		fprintf(figout,'    };\n\n');	
	end
	fprintf(figout,'\n\n');		
 	fprintf(figout,'    \\draw (axis cs:%e,%e)coordinate(plot4all) -- ',S(1),S(3));
 	fprintf(figout,'(axis cs: %e,%e)coordinate(plot4alr) -- ',S(2),S(3));
 	fprintf(figout,'(axis cs: %e,%e)coordinate(plot4aur) --',S(2),S(4)); 
 	fprintf(figout,'(axis cs: %e,%e)coordinate(plot4aul) -- cycle; %%\n',S(1),S(4));
 	fprintf(figout,'\n\n');		
	fprintf(figout,'    \\legend{');
	for ii=1:md
		if (mod(ii-1,10)==1)
			if (mod(ii-1,100)==11)
				fprintf(figout,'%dth order approximation,',ii-1);
			else
				fprintf(figout,'%dst order approximation,',ii-1);
			end
		elseif (mod(ii-1,10)==2)
			if (mod(ii-1,100)==12)
				fprintf(figout,'%dth order approximation,',ii-1);
			else
				fprintf(figout,'%dnd order approximation,',ii-1);
			end
		elseif (mod(ii-1,10)==3)
			if (mod(ii-1,100)==13)
				fprintf(figout,'%dth order approximation,',ii-1);
			else
				fprintf(figout,'%drd order approximation,',ii-1);
			end
		else
			fprintf(figout,'%dth order approximation,',ii-1);
		end
	end		
	fprintf(figout,'}%%\n',md-1);    
	fprintf(figout,'	\\end{semilogyaxis}%%\n');
	
 	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 	% magnified plot
 	fprintf(figout,'  \\begin{semilogyaxis}[ %%\n');
 	fprintf(figout,'    remember picture, %%\n');
 	fprintf(figout,'    name = plot4b, %%\n');
 	fprintf(figout,'    at = {($(plot4a.north east)+(0.5cm,1cm)$)}, %%\n');
 	fprintf(figout,'    anchor = north east, %%\n');
 	fprintf(figout,'    scale only axis, %%\n');
 	fprintf(figout,'    scaled y ticks = false,%%\n');
 	fprintf(figout,'    width=0.30\\textwidth,%%\n');
 	fprintf(figout,'    height=0.14\\textheight,%%\n');
 	fprintf(figout,'    xmin = %e, xmax = %e, %% change if needed\n', S(1), S(2));
 	fprintf(figout,'    ymin = %d, ymax = %d, %%\n', S(3) ,S(4));
 	fprintf(figout,'    y tick label style= {anchor=west,xshift=3pt},%%\n');
 	fprintf(figout,'    x tick label style= {anchor=south,yshift=3pt},%%\n');
	fprintf(figout,'    ytick={1e-15,1e-14,1e-13}, %%\n');	
	fprintf(figout,'    xtick={0,0.1,0.2,0.3}, %%\n');	
 	fprintf(figout,'    scaled ticks = false,%%\n');
	fprintf(figout,'    axis on top, %%\n');
	fprintf(figout,'    every axis post/.style={thick, no marks},%%\n');
 	fprintf(figout,'    mark size = 2, %%\n');
 	fprintf(figout,'	  axis background/.style = {fill=white,opacity = 1}]\n\n');
 	

 	minkk = sum(xl<S(1))+1;
 	maxkk = sum(xl<=S(2));
 	
 	for ii=1:md
 		printed = false;
 		el = zeros(1,n); 
 		el(ii) = 1;
 		for kk = minkk:maxkk
 			if ((z(kk,ii)<=S(4)) && (z(kk,ii)>=S(3)))
 				if (~printed)
 					fprintf(figout,'    \\addplot\n');
 					fprintf(figout,'    coordinates{%%\n');
 					printed = true;
 				end
 				fprintf(figout,'    (%e,%e)%%\n',xl(kk),z(kk,ii));
			end
 		end			
		if (printed)
 			fprintf(figout,'    };\n\n');
 		end
 	end
 	fprintf(figout,'	\\end{semilogyaxis}%%\n');
	
	
 	fprintf(figout,'	\\draw (current axis.north west)--(plot4aul); %%\n');
 	fprintf(figout,'	\\draw (current axis.south east)--(plot4alr); %%\n');

	fprintf(figout,'\\end{tikzpicture}%%\n');
	fprintf(figout,'\\endpgfgraphicnamed%%\n');
	fprintf(figout,'\\caption{Example %d---Maximal difference between Taylor approximations and eigenvalues.}%%\n',example);
	fprintf(figout,'\\label{fig:example4_%d}%%\n',example);
	fprintf(figout,'\\end{figure}\n\n\n');
	fprintf(figout,'%%%% Thomas'' emacs local variable set up\n\n');
	fprintf(figout,'%%%%%% Local Variables:\n'); 
	fprintf(figout,'%%%%%% mode: LaTeX\n');
	fprintf(figout,'%%%%%% TeX-master: "pevp"\n');
	fprintf(figout,'%%%%%% TeX-PDF-mode:t\n');
	fprintf(figout,'%%%%%% TeX-engine: luatex\n');
	fprintf(figout,'%%%%%% auto-fill-function:nil\n');
	fprintf(figout,'%%%%%% mode:auto-fill\n');
	fprintf(figout,'%%%%%% flyspell-mode:nil\n');
	fprintf(figout,'%%%%%% mode:flyspell\n');
	fprintf(figout,'%%%%%% ispell-local-dictionary: "american"\n');
	fprintf(figout,'%%%%%% End: \n');

	fclose(figout);

	
	if (~exist('h2','var'))
		h2 = figure;
		set(h2,'WindowStyle','docked');
	else
		figure(h2);
	end
	semilogy(xl,abs(z));
	xlim(T);
	legend('0','1','2','3','4','5','6','7','8','9','Location','northeastoutside');

	title('eigenvalues')
	
	if (~exist('h2b','var'))
		h2b = figure;
		set(h2b,'WindowStyle','docked');
	else
		figure(h2b);
	end
	semilogy(xl,abs(zv));
	xlim(T);
	legend('0','1','2','3','4','5','6','7','8','9','Location','northeastoutside');

	title('eigenvector 2nd largest eigenvalue')
	

	if (~exist('h3','var'))
		h3 = figure;
		set(h3,'WindowStyle','docked');
	else
		figure(h3);
	end
	
	plot(store_mu_rand','+')
	title('sampled mu');

		if (~exist('h4','var'))
		h4 = figure;
		set(h4,'WindowStyle','docked');
	else
		figure(h4);
	end
	
	plot(store_sampled_eig','+')
	title('sampled 2nd and 3rd largest eigenvalue (Taylor polynomial)');

	if (~exist('h5','var'))
		h5 = figure;
		set(h5,'WindowStyle','docked');
	else
		figure(h5);
	end
	
	plot(store_sampled_eig_exact','+')
	title('sampled 2nd and 3rd largest eigenvalue (exact)');

	if (~exist('h6','var'))
		h6 = figure;
		set(h6,'WindowStyle','docked');
	else
		figure(h6);
	end
	semilogy(transpose(abs(store_sampled_eig-store_sampled_eig_exact)),'+')
	title('error in sampled eigenvalues');
	
	if (~exist('h9','var'))
		h9 = figure;
		set(h9,'WindowStyle','docked');
	else
		figure(h9);
	end
	semilogy(transpose(error_eig_vec),'+')
	title('error in sampled eigenvectors');


	if (~exist('h7','var'))
		h7 = figure;
		set(h7,'WindowStyle','docked');
	else
		figure(h7);
	end
	mbin = ceil(max(max(store_sampled_eig_exact)))+2;
	histogram(store_sampled_eig(1,:),0:1:mbin)
	hold on
	histogram(store_sampled_eig(2,:),0:1:mbin)
	hold off
	title('histogram 2nd and 3rd largest eigenvalue (Taylor polynomial)');

	if (~exist('h8','var'))
		h8 = figure;
		set(h8,'WindowStyle','docked');
	else
		figure(h8);
	end
	
	histogram(store_sampled_eig_exact(1,:),0:1:mbin)
	hold on
	histogram(store_sampled_eig_exact(2,:),0:1:mbin)
	hold off
	title('histogram 2nd and 3rd largest eigenvalue (exact)');

	
	if (~exist('h10','var'))
		h10 = figure;
		set(h10,'WindowStyle','docked');
	else
		figure(h10);
	end

	plot3(proj_ev(:,1),proj_ev(:,2),proj_ev(:,3),'b+')
	hold on
	plot3(proj_ev(:,4),proj_ev(:,5),proj_ev(:,6),'r+')
	plot3(proj_ev_exact(:,4),proj_ev_exact(:,5),proj_ev_exact(:,6),'rx')
	plot3(proj_ev_exact(:,1),proj_ev_exact(:,2),proj_ev_exact(:,3),'bx')
	
	title('Plot of Eigenvectors in the space spanned by the eigenvectors 1,2, and 3 for mu=t0');

	
end

end



%%% Local Variables: 
%%% mode:matlab
%%% flyspell-mode:nil
%%% mode:flyspell-prog
