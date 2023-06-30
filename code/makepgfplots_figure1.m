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
		filename = sprintf('exp_%s_figure41_chebyshev_%d_%d_%f_%f_%d.tex',type,n,p,T2(1),T2(2),example);
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
%fprintf(figout,'    set layers=axis on top,%%\n');
fprintf(figout,'    scaled y ticks = false,%%\n');
fprintf(figout,'    scale ticks below exponent ={-3},%%\n');
if (type=='tal')	
	%fprintf(figout,'    clip mode = individual,%%\n');
	fprintf(figout,'    scale only axis, %%\n');
	fprintf(figout,'    width=120mm,%%\n');
	fprintf(figout,'    height=60mm,%%\n');
	if (n==8)&&(p==7)&&(t0==0.025)
		fprintf(figout,'    ytick = {0,1,2,4,6,8},%%\n');
		fprintf(figout,'    yticklabels = {$0$,{\\color{white}$10^{-12}$},$2$,$4$,$6$,$8$},%%\n');
	end
else
	fprintf(figout,'    scale only axis, %%\n');
	fprintf(figout,'    width=0.75\\textwidth,%%\n');
	fprintf(figout,'    height=0.22067\\textheight,%%\n');
end
fprintf(figout,'    xmin = %e, xmax = %e, %% change if needed\n', T(1), T(2));
if (example==5)
	fprintf(figout,'    ymin = %e, ymax = %e, %%\n', -10, 0);	
	fprintf(figout,'    restrict y to domain=%d:%d, %%\n',-12,2);
else	
	fprintf(figout,'    ymin = %e, ymax = %e, %%\n', 0, n);
	fprintf(figout,'    restrict y to domain=%d:%d, %%\n',-2,n+2);
end
fprintf(figout,'    restrict x to domain=%d:%d, %%\n', floor(T(1))-2, ceil(T(2))+2);
fprintf(figout,'    xlabel = {$\\mu$},%%\n');
fprintf(figout,'    ylabel = {Eigenvalues},%%\n');
fprintf(figout,'    scaled ticks = false,%%\n');
fprintf(figout,'    every axis y label/.style= {at={(ticklabel cs:0.5)},rotate=90,yshift=3pt},%%\n');
fprintf(figout,'    tick label style={/pgf/number format/fixed},%%\n'); 
fprintf(figout,'    axis on top, %%\n');
if (type=='tal')	
	fprintf(figout,'    every axis legend/.append style={ at={(0.00,1.02)}, anchor=south west,\n');
else
	fprintf(figout,'    every axis legend/.append style={ at={(0.00,1.05)}, anchor=south west,\n');
end
fprintf(figout,'      cells={anchor=west}, },%%\n');
fprintf(figout,'    mark size = 2]\n\n');

% draw \mu0
fprintf(figout,'    \\draw[SPECgreen,thick] (axis cs:%e,%e) --',t0,0);
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
			
		case {5}
			for ii = 1:n
				for jj = 1:n
					Ax(ii,jj) = (M{ii,jj}(xx)); 
				end
			end

			
	end
	
	e = eig(Ax);
	%plot(xx*ones(n,1),e,'r+');
	for ii=1:n
		fprintf(figout,'    (%e,%e)%%\n',xx,e(ii));
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
				fprintf(figout,'    (%e,%e)%%\n',xl(kk),hhorn);
			end
			
		else
			hfp = el*fp{p}(xl(kk));
			if ((hfp<=80) && (hfp>=-10))

				fprintf(figout,'    (%e,%e)%%\n',xl(kk),hfp);
			end
			
		end
	end			
	fprintf(figout,'    };\n\n');	
end

if (S ~= [42,42,42,42])
	fprintf(figout,'\n\n');		
	fprintf(figout,'    \\draw (axis cs:%e,%e)coordinate(plot1all) -- ',S(1),S(3));
	fprintf(figout,'(axis cs:%e,%e)coordinate(plot1alr) -- ',S(2),S(3));
	fprintf(figout,'(axis cs: %e,%e)coordinate(plot1aur) --',S(2),S(4)); 
	fprintf(figout,'(axis cs: %e,%e)coordinate(plot1aul) -- cycle; %%\n',S(1),S(4));
	fprintf(figout,'\n\n');		
end

if (taylor)
	fprintf(figout,'    \\legend{Eigenvalues, %dth Taylor approximation}%%\n',p-1);    
else
	fprintf(figout,'    \\legend{Eigenvalues, %dth Chebyshev approximation}%%\n',p-1);    
end	

fprintf(figout,'	\\end{axis}%%\n');


if (S ~= [42,42,42,42])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% magnified plot
fprintf(figout,'  \\begin{axis}[ %%\n');
fprintf(figout,'    remember picture, %%\n');
fprintf(figout,'    name = plot1b, %%\n');
fprintf(figout,'    at = {($(plot1a.north east)+(0.5cm,1cm)$)}, %%\n');
fprintf(figout,'    anchor = north east, %%\n');
%fprintf(figout,'    set layers=axis on top,%%\n');
fprintf(figout,'    scaled y ticks = false,%%\n');
if (type=='tal')
	%fprintf(figout,'    clip mode = individual,%%\n');
	fprintf(figout,'    scale only axis, %%\n');
	fprintf(figout,'    width=50mm,%%\n');
	fprintf(figout,'    height=40mm,%%\n');
else
	fprintf(figout,'    scale only axis, %%\n');
	fprintf(figout,'    width=0.30\\textwidth,%%\n');
	fprintf(figout,'    height=0.14\\textheight,%%\n');
end
fprintf(figout,'    xmin = %e, xmax = %e, %% change if needed\n', S(1), S(2));
fprintf(figout,'    ymin = %d, ymax = %d, %%\n', S(3) ,S(4));
fprintf(figout,'    scaled ticks = false,%%\n');
fprintf(figout,'    y tick label style= {anchor=east,xshift=-3pt},%%\n');
fprintf(figout,'    x tick label style= {anchor=north,yshift=-3pt},%%\n');
fprintf(figout,'    tick label style={/pgf/number format/fixed},%%\n'); 
if (type=='pap')
	fprintf(figout,'    ytick = {1.4}, %%\n');
	fprintf(figout,'    xtick = {0.2,0.4}, %%\n');
end
fprintf(figout,'    axis on top, %%\n');
fprintf(figout,'    mark size = 2, %%\n');
fprintf(figout,'	  axis background/.style = {fill=white,opacity = 1}]\n\n');

% draw \mu0
fprintf(figout,'    \\draw[SPECgreen,thick] (axis cs:%e,%e) --',t0,S(3));
fprintf(figout,'    (axis cs:%e,%e);\n\n',t0,S(4));

fprintf(figout,'	  \\addplot[SPECred, only marks, mark=+]\n');
fprintf(figout,'    coordinates{%%\n');      


minkk = sum(xl<S(1))+1;
maxkk = sum(xl<=S(2));
for kk = minkk:maxkk
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
		if ((e(ii)<=S(4)) && (e(ii)>=S(3)))
			fprintf(figout,'    (%e,%e)%%\n',xx,e(ii));
		end
	end
end
fprintf(figout,'    };\n\n');	


for ii=1:n
	printed = false;
	el = zeros(1,n); 
	el(ii) = 1;
	
	fprintf(figout,'    \\addplot[SPECblue, thick]\n');
	fprintf(figout,'    coordinates{%%\n');
	
	for kk = minkk:maxkk
		
		if (taylor)
			if ((el*horner_f(xl(kk),t0,dp(:,1:p))<=S(4)) && (el*horner_f(xl(kk),t0,dp(:,1:p))>=S(3)))||(kk==minkk)			
				fprintf(figout,'    (%e,%e)%%\n',xl(kk),el*horner_f(xl(kk),t0,dp(:,1:p)));
			
			end				
			
		else
			
			if ((el*fp{p}(xl(kk))<=S(4)) && (el*fp{p}(xl(kk))>=S(3)))||(kk==minkk)
				fprintf(figout,'    (%e,%e)%%\n',xl(kk),el*fp{p}(xl(kk)));
				
			end
		end
	end			
	fprintf(figout,'    };\n\n');
end
fprintf(figout,'	\\end{axis}%%\n');


fprintf(figout,'	\\draw (current axis.north west)--(plot1aul); %%\n');
fprintf(figout,'	\\draw (current axis.south east)--(plot1alr); %%\n');
end

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
if (type=='tal')
	fprintf(figout,'%%%%%% TeX-master: "parametric_evp"\n');
else
	fprintf(figout,'%%%%%% TeX-master: "pevp2"\n');
end
fprintf(figout,'%%%%%% TeX-PDF-mode:t\n');
fprintf(figout,'%%%%%% TeX-engine: luatex\n');
fprintf(figout,'%%%%%% auto-fill-function:nil\n');
fprintf(figout,'%%%%%% mode:auto-fill\n');
fprintf(figout,'%%%%%% flyspell-mode:nil\n');
fprintf(figout,'%%%%%% mode:flyspell\n');
fprintf(figout,'%%%%%% ispell-local-dictionary: "american"\n');
fprintf(figout,'%%%%%% End: \n');

fclose(figout);
	
