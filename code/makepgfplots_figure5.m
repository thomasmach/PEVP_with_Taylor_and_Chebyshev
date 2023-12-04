%% Plot of sampled eigenvalues and Taylor/Chebushev approximation
% this file produces a tex-file that is directly inserted in the paper

fprintf('Running makepgfplots_figure1.m; type: %s\n',type);

if (taylor)
	
	if (usevpa)
		filename = sprintf('exp_%s_figure41_taylor_%d_%d_%f_%d_vpa.tex',type,n,p,t0,example);
	else
		filename = sprintf('exp_%s_figure41_taylor_%d_%d_%f_%d.tex',type,n,p,t0,example);
	end

else
	
	if (usevpa)
		filename = sprintf('exp_%s_figure41_chebyshev_%d_%d_%f_%f_%d_vpa.tex',type,n,p,T2(1),T2(2),example);
	else
		filename = sprintf('exp_%s_figure41_chebyshev_%d_%d_%f_%f_%d.tex',n,type,p,T2(1),T2(2),example);
	end
	
end
fprintf('Writing file %s\n',filename);
figout = fopen(filename,'w');


if (~taylor)

	fp{1} = @(mu) dp(:,1)*Usch{1}(mu);
	fvp{1} = @(mu) vp(:,:,1)*Usch{1}(mu);

	for kk = 2:p
		fp{kk} = @(mu) fp{kk-1}(mu) + dp(:,kk)*Usch{kk}(mu);
		fvp{kk} = @(mu) fvp{kk-1}(mu) + vp(:,:,kk)*Usch{kk}(mu);
	end

end

	
if (type=='pap')
	fprintf(figout,'\\begin{figure}\n');
	fprintf(figout,'\\centering\n');
end
fprintf(figout,'\\beginpgfgraphicnamed{figure-%d-%d-%d}\n',n,p,round(t0*100));
fprintf(figout,'\\begin{tikzpicture}\n');
fprintf(figout,'  \\begin{axis}[ %%\n');
fprintf(figout,'    remember picture, %%\n');
fprintf(figout,'    name = plot1a, %%\n');
fprintf(figout,'    scale only axis, %%\n');
fprintf(figout,'    xticklabel pos=right, %%\n');
if (type=='tal')	
	%fprintf(figout,'    clip mode = individual,%%\n');
	%fprintf(figout,'    set layers=axis on top,%%\n');
	fprintf(figout,'    scale only axis, %%\n');
	fprintf(figout,'    width=120mm,%%\n');
	fprintf(figout,'    height=30mm,%%\n');
	if (n==8)&&(p==7)&&(t0==0.025)
		fprintf(figout,'    ytick = {0,1,2,4,6,8},%%\n');
		fprintf(figout,'    yticklabels = {$0$,{\\color{white}$10^{-12}$},$2$,$4$,$6$,$8$},%%\n');
	end
else
	fprintf(figout,'    scaled y ticks = false,%%\n');
	fprintf(figout,'    width=0.75\\textwidth,%%\n');
	%fprintf(figout,'    height=0.22067\\textheight,%%\n');
	fprintf(figout,'    height=0.11\\textheight,%%\n');
end
fprintf(figout,'    xmin = %e, xmax = %e, %% change if needed\n', T(1), T(2));
fprintf(figout,'    ymin = %e, ymax = %e, %%\n', -0.5, 4.5);
fprintf(figout,'    restrict x to domain=%d:%d, %%\n', floor(T(1))-2, ceil(T(2))+2);
fprintf(figout,'    restrict y to domain=%d:%d, %%\n',-20,20);
fprintf(figout,'    xlabel = {$\\mu$},%%\n');
fprintf(figout,'    ylabel = {real$(\\lambda(\\mu))$},%%\n');
fprintf(figout,'    scaled ticks = false,%%\n');
fprintf(figout,'    every axis y label/.style= {at={(ticklabel cs:0.5)},rotate=90,yshift=3pt},%%\n');
fprintf(figout,'    axis on top, %%\n');
fprintf(figout,'    every axis legend/.append style={ at={(0.02,0.975)}, anchor=north west,\n');
fprintf(figout,'      cells={anchor=west}, },%%\n');
fprintf(figout,'    mark size = 2]\n\n');

% draw \mu0
fprintf(figout,'    \\draw[SPECgreen,thick] (axis cs:%e,%e) --',t0,-n);
fprintf(figout,'    (axis cs:%e,%e) node[midway, right]{$\\mu_{0}$};\n\n',t0,n);

fprintf(figout,'	  \\addplot[SPECred, only marks, mark=+]\n');
fprintf(figout,'    coordinates{%%\n');      


for kk = 1:npoints
	xx = xl(kk);
	
	switch (example)
		case {1,111}
			Ax(:,:) = exp(-xx*U);
			
		case {2,112}
			Ax(:,:) = diag([ones(floor(n/2),1);(1/xx)*ones(2,1);ones(n-floor(n/2)-2,1)])*K;
			
		case {3,113}
			Ax(:,:) = diag(ones(n,1),0) + diag(ones(n-1,1),1);
			Ax(n,1) = xx;
			
	end
	
	e = eig(Ax);
	%plot(xx*ones(n,1),e,'r+');
	for ii=1:n
		fprintf(figout,'    (%e,%e)%%\n',xx,real(e(ii)));
	end
		
end
fprintf(figout,'    };\n\n');	


for ii=1:n
	fprintf(figout,'    \\addplot[SPECblue, thick]\n');
	fprintf(figout,'    coordinates{%%\n');
	el = zeros(1,n); 
	el(ii) = 1;
	for kk = 1:npoints
		if (taylor)
			hhorn = el*horner_f(xl(kk),t0,dp(:,1:p)); 
			if ((hhorn<=80) && (hhorn>=-10))
				fprintf(figout,'    (%e,%e)%%\n',xl(kk),real(hhorn));
			end
			
		else
			hfp = el*fp{p}(xl(kk));
			if ((hfp<=80) && (hfp>=-10))

				fprintf(figout,'    (%e,%e)%%\n',xl(kk),real(hfp));
			end
			
		end
	end			
	fprintf(figout,'    };\n\n');	
end

if (taylor)
	fprintf(figout,'    \\legend{Eigenvalues, %dth Taylor approximation}%%\n',p-1);    
else
	fprintf(figout,'    \\legend{Eigenvalues, %dth Chebyshev approximation}%%\n',p-1);    
end	

fprintf(figout,'	\\end{axis}%%\n');


fprintf(figout,'  \\begin{axis}[ %%\n');
fprintf(figout,'    at ={($(plot1a.south)+(0,3pt)$)}, %%\n');
fprintf(figout,'    anchor = north, %%\n');
fprintf(figout,'    remember picture, %%\n');
fprintf(figout,'    name = plot1b, %%\n');
if (type=='tal')	
	%fprintf(figout,'    clip mode = individual,%%\n');
	%fprintf(figout,'    set layers=axis on top,%%\n');
	fprintf(figout,'    scale only axis, %%\n');
	fprintf(figout,'    width=120mm,%%\n');
	fprintf(figout,'    height=30mm,%%\n');
	if (n==8)&&(p==7)&&(t0==0.025)
		fprintf(figout,'    ytick = {0,1,2,4,6,8},%%\n');
		fprintf(figout,'    yticklabels = {$0$,{\\color{white}$10^{-12}$},$2$,$4$,$6$,$8$},%%\n');
	end
else
	fprintf(figout,'    scaled y ticks = false,%%\n');
	fprintf(figout,'    width=0.75\\textwidth,%%\n');
	%fprintf(figout,'    height=0.22067\\textheight,%%\n');
	fprintf(figout,'    height=0.11\\textheight,%%\n');
end
fprintf(figout,'    xmin = %e, xmax = %e, %% change if needed\n', T(1), T(2));
fprintf(figout,'    ymin = %e, ymax = %e, %%\n', -1.75, 1.75);
fprintf(figout,'    restrict x to domain=%d:%d, %%\n', floor(T(1))-2, ceil(T(2))+2);
fprintf(figout,'    restrict y to domain=%d:%d, %%\n',-20,20);
fprintf(figout,'    xlabel = {$\\mu$},%%\n');
%fprintf(figout,'    ylabel = {imag(Eigenvalues)},%%\n');
fprintf(figout,'    ylabel = {imag$(\\lambda(\\mu))$},%%\n');
fprintf(figout,'    scaled ticks = false,%%\n');
fprintf(figout,'    every axis y label/.style= {at={(ticklabel cs:0.5)},rotate=90,yshift=3pt},%%\n');
fprintf(figout,'    axis on top, %%\n');
fprintf(figout,'    every axis legend/.append style={ at={(0.00,1.05)}, anchor=south west,\n');
fprintf(figout,'      cells={anchor=west}, },%%\n');
fprintf(figout,'    mark size = 2]\n\n');

% draw \mu0
fprintf(figout,'    \\draw[SPECgreen,thick] (axis cs:%e,%e) --',t0,-n);
fprintf(figout,'    (axis cs:%e,%e) node[midway, right]{$\\mu_{0}$};\n\n',t0,n);

fprintf(figout,'	  \\addplot[SPECred, only marks, mark=+]\n');
fprintf(figout,'    coordinates{%%\n');      


for kk = 1:npoints
	xx = xl(kk);
	
	switch (example)
		case {1,111}
			Ax(:,:) = exp(-xx*U);
			
		case {2,112}
			Ax(:,:) = diag([ones(floor(n/2),1);(1/xx)*ones(2,1);ones(n-floor(n/2)-2,1)])*K;
			
		case {3,113}
			Ax(:,:) = diag(ones(n,1),0) + diag(ones(n-1,1),1);
			Ax(n,1) = xx;
			
	end
	
	e = eig(Ax);
	%plot(xx*ones(n,1),e,'r+');
	for ii=1:n
		fprintf(figout,'    (%e,%e)%%\n',xx,imag(e(ii)));
	end
		
end
fprintf(figout,'    };\n\n');	


for ii=1:n
	fprintf(figout,'    \\addplot[SPECblue, thick]\n');
	fprintf(figout,'    coordinates{%%\n');
	el = zeros(1,n); 
	el(ii) = 1;
	for kk = 1:npoints
		if (taylor)
			hhorn = el*horner_f(xl(kk),t0,dp(:,1:p)); 
			if ((hhorn<=80) && (hhorn>=-10))
				fprintf(figout,'    (%e,%e)%%\n',xl(kk),imag(hhorn));
			end
			
		else
			hfp = el*fp{p}(xl(kk));
			if ((hfp<=80) && (hfp>=-10))

				fprintf(figout,'    (%e,%e)%%\n',xl(kk),imag(hfp));
			end
			
		end
	end			
	fprintf(figout,'    };\n\n');	
end

fprintf(figout,'	\\end{axis}%%\n');











fprintf(figout,'\\end{tikzpicture}%%\n');
fprintf(figout,'\\endpgfgraphicnamed%%\n');

if (type=='pap')

	if (taylor)
		fprintf(figout,'\\caption{Example~\\ref{example:%d}, $n=%d$, $\\mu_{0}=%3.2f$.}%%\n',example,n,t0);
	else
		fprintf(figout,'\\caption{Example~\\ref{example:%d}, $n=%d$, $[\\mu_{1},\\mu_{2}]=[%3.2f,%3.2f]$.}%%\n',example,n,T2(1),T2(2));
	end
	
	if (taylor)
		if (usevpa)||(usesingle)
			fprintf(figout,'\\label{fig:example1vpa_%d_%d_%d_%3.2f_%d}%%\n',taylor,n,p,t0,example);
		else
			fprintf(figout,'\\label{fig:example1_%d_%d_%d_%3.2f_%d}%%\n',taylor,n,p,t0,example);
		end		
	else	
		if (usevpa)||(usesingle)
			fprintf(figout,'\\label{fig:example1vpa_%d_%d_%d_%3.2f_%d}%%\n',taylor,n,p,T2(1),example);
		else
			fprintf(figout,'\\label{fig:example1_%d_%d_%d_%3.2f_%d}%%\n',taylor,n,p,T2(1),example);
		end		
	end
	
	fprintf(figout,'\\end{figure}\n\n\n');
end
fprintf(figout,'%%%% Thomas'' emacs local variable set up\n\n');
fprintf(figout,'%%%%%% Local Variables:\n'); 
fprintf(figout,'%%%%%% mode: LaTeX\n');
fprintf(figout,'%%%%%% TeX-master: "pevp2"\n');
fprintf(figout,'%%%%%% TeX-PDF-mode:t\n');
fprintf(figout,'%%%%%% TeX-engine: luatex\n');
fprintf(figout,'%%%%%% auto-fill-function:nil\n');
fprintf(figout,'%%%%%% mode:auto-fill\n');
fprintf(figout,'%%%%%% flyspell-mode:nil\n');
fprintf(figout,'%%%%%% mode:flyspell\n');
fprintf(figout,'%%%%%% ispell-local-dictionary: "american"\n');
fprintf(figout,'%%%%%% End: \n');

fclose(figout);
	
