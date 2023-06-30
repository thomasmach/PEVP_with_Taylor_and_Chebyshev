%% Plot of maximal difference betw. Taylor/Chebushev and smapled eigenvalues
% this file produces a tex-file that is directly inserted in the paper

fprintf('Running makepgfplots_figure2.m; type: %s\n',type);

if (usevpa)||(usesingle)
	filename = sprintf('exp_%s_fig52_%d_%d_%f_%d_vpa.tex',type,n,p,t0,example);
else		
	if (~exist('fnns','var')||fnns)
		filename = sprintf('exp_%s_fig52_%d_%d_%f_%d_%d.tex',type,n,p,t0,example,newtonsteps);
	else	
		filename = sprintf('exp_%s_fig52_%d_%d_%f_%d.tex',type,n,p,t0,example);
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
if (~exist('fnns','var')||fnns)
	fprintf(figout,'\\beginpgfgraphicnamed{figure2-%d-%d-%d-%d}\n',n,p,round(t0*100),newtonsteps);
else
	fprintf(figout,'\\beginpgfgraphicnamed{figure2-%d-%d-%d}\n',n,p,round(t0*100));
end
fprintf(figout,'\\begin{tikzpicture}\n');
fprintf(figout,'  \\begin{semilogyaxis}[ %%\n');
fprintf(figout,'    remember picture, %%\n');
fprintf(figout,'    name = plot4a, %%\n');
fprintf(figout,'    scale only axis, %%\n');
fprintf(figout,'    scaled y ticks = false,%%\n');
fprintf(figout,'    scale ticks below exponent ={-3},%%\n');
if (type=='tal')	
	%fprintf(figout,'    clip mode = individual,%%\n');
	fprintf(figout,'    width=120mm,%%\n');
	fprintf(figout,'    height=60mm,%%\n');
else
	fprintf(figout,'    width=0.75\\textwidth,%%\n');
	fprintf(figout,'    height=0.22067\\textheight,%%\n');
end
fprintf(figout,'    xmin = %e, xmax = %e, %% change if needed\n', T(1), T(2));
fprintf(figout,'    ymin = %e, ymax = %e, %%\n', 1e-18, 1e6);
fprintf(figout,'    ytick={1e-16,1e-12,1e-8,1e-4,1e0,1e4}, %%\n');
fprintf(figout,'    xlabel = {$\\mu$},%%\n');
fprintf(figout,'    ylabel = {Error},%%\n');
fprintf(figout,'    scaled ticks = false,%%\n');
fprintf(figout,'    every axis y label/.style= {at={(ticklabel cs:0.5)},rotate=90,yshift=3pt},%%\n');
fprintf(figout,'    tick label style={/pgf/number format/fixed},%%\n'); 
fprintf(figout,'    axis on top, %%\n');
%fprintf(figout,'    restrict y to domain=%d:%d, %%\n', 1e-18, 1e6);
fprintf(figout,'    every axis legend/.append style={ at={(0.00,1.05)}, anchor=south west,\n');
fprintf(figout,'      cells={anchor=west}, legend columns=2},%%\n');
fprintf(figout,'    every axis post/.style={thick, no marks},%%\n');
fprintf(figout,'    cycle list name = warmercolors,%%\n');
fprintf(figout,'    mark size = 2]\n\n');

if (taylor)
	% draw \mu0
	fprintf(figout,'    \\draw[SPECred,thick] (axis cs:%e,%e) --',t0,1e-18);
	fprintf(figout,'    (axis cs:%e,%e) node[midway, right]{$\\mu_{0}$};\n\n',t0,1e6);
end

zzz = zeros(npoints,p);
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
	if (taylor)

		for ll = 1:p
			zzz(kk,ll) = norm(sort(e)-sort(horner_f(xx,t0,dp(:,1:ll))));
		end
	
	else
		
		for ll = 1:p
			zzz(kk,ll) = norm(sort(e)-sort(fp{ll}(xx)));
% 			for tt = 1:n
% 				et = zeros(n,1);
% 				et(tt) = 1;
% 				ze(kk,ll,tt) = min(abs(e-et'*sort(fp{ll}(xx))));
% 			end
		end
		
	end
end



for ii=1:p
	fprintf(figout,'    \\addplot\n');
	fprintf(figout,'    coordinates{%%\n');
	el = zeros(1,n); 
	el(ii) = 1;
	for kk = 1:npoints
		if (abs(zzz(kk,ii))<1e6)||(kk==1)
			fprintf(figout,'    (%e,%e)%%\n',xl(kk),zzz(kk,ii));
		end			
	end
	fprintf(figout,'    };\n\n');	
end

indmin = min(find(xl>t0));
if (usesingle)
	indlast = min(find(zzz(indmin:end,p)>1e2));
	ind15 = min(find(zzz(indmin:end,15)>1e-6))-1;
else
	indlast = min(find(zzz(indmin:end,p)>1e-13));
end

fprintf(figout,'    \\node [coordinate,pin=above:{0th order}] at (axis cs:%e,%e){};\n',xl(indmin),zzz(indmin,1));		
if (usesingle)
	
	fprintf(figout,'    \\node [coordinate,pin=left:{%dth order}] at (axis cs:%e,%e){};\n',p-1,xl(indlast+indmin),zzz(indlast+indmin,p));		
	fprintf(figout,'    \\node [coordinate,pin=right:{%dth order}] at (axis cs:%e,%e){};\n',15-1,xl(ind15+indmin),zzz(ind15+indmin,15));		

else
	
	if (taylor)
		fprintf(figout,'    \\node [coordinate,pin=right:{%dth order}] at (axis cs:%e,%e){};\n',p-1,xl(indlast+indmin),zzz(indlast+indmin,p));		
	else
		fprintf(figout,'    \\node [coordinate,pin=below:{%dth order}] at (axis cs:%e,%e){};\n',p-1,xl(indlast+indmin),zzz(indlast+indmin,p));		
	end
end

if (S ~= [42,42,42,42])
	fprintf(figout,'\n\n');		
	fprintf(figout,'    \\draw (axis cs:%e,%e)coordinate(plot4all) -- ',S(1),S(3));
	fprintf(figout,'(axis cs: %e,%e)coordinate(plot4alr) -- ',S(2),S(3));
	fprintf(figout,'(axis cs: %e,%e)coordinate(plot4aur) --',S(2),S(4)); 
	fprintf(figout,'(axis cs: %e,%e)coordinate(plot4aul) -- cycle; %%\n',S(1),S(4));
	fprintf(figout,'\n\n');		
end

%fprintf(figout,'    \\legend{');
%for ii=1:p
%	if (mod(ii-1,10)==1)
%		if (mod(ii-1,100)==11)
%			fprintf(figout,'%dth order approximation,',ii-1);
%		else
%			fprintf(figout,'%dst order approximation,',ii-1);
%		end
%	elseif (mod(ii-1,10)==2)
%		if (mod(ii-1,100)==12)
%			fprintf(figout,'%dth order approximation,',ii-1);
%		else
%			fprintf(figout,'%dnd order approximation,',ii-1);
%		end
%	elseif (mod(ii-1,10)==3)
%		if (mod(ii-1,100)==13)
%			fprintf(figout,'%dth order approximation,',ii-1);
%		else
%			fprintf(figout,'%drd order approximation,',ii-1);
%		end
%	else
%		fprintf(figout,'%dth order approximation,',ii-1);
%	end
%end		
%fprintf(figout,'}%%\n',p-1);    
fprintf(figout,'	\\end{semilogyaxis}%%\n');

if (S ~= [42,42,42,42])
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% magnified plot
fprintf(figout,'  \\begin{semilogyaxis}[ %%\n');
fprintf(figout,'    remember picture, %%\n');
fprintf(figout,'    name = plot4b, %%\n');
fprintf(figout,'    at = {($(plot4a.north east)+(0.5cm,1.2cm)$)}, %%\n');
fprintf(figout,'    anchor = north east, %%\n');
fprintf(figout,'    scale only axis, %%\n');
fprintf(figout,'    scaled y ticks = false,%%\n');
if (type=='tal')
	%fprintf(figout,'    clip mode = individual,%%\n');
	fprintf(figout,'    width=50mm,%%\n');
	fprintf(figout,'    height=40mm,%%\n');
else
	fprintf(figout,'    width=0.30\\textwidth,%%\n');
	fprintf(figout,'    height=0.14\\textheight,%%\n');
end
fprintf(figout,'    xmin = %e, xmax = %e, %% change if needed\n', S(1), S(2));
fprintf(figout,'    ymin = %d, ymax = %d, %%\n', S(3) ,S(4));
fprintf(figout,'    y tick label style= {anchor=east,xshift=-3pt},%%\n');
fprintf(figout,'    x tick label style= {anchor=north,yshift=-3pt},%%\n');
%fprintf(figout,'    ytick={1e-15,1e-14,1e-13}, %%\n');	
%fprintf(figout,'    xtick={0,0.1,0.2,0.3}, %%\n');	
fprintf(figout,'    scaled ticks = false,%%\n');
fprintf(figout,'    tick label style={/pgf/number format/fixed},%%\n'); 
fprintf(figout,'    axis on top, %%\n');
fprintf(figout,'    every axis post/.style={thick, no marks},%%\n');
fprintf(figout,'    cycle list name = warmercolors,%%\n');
fprintf(figout,'    mark size = 2, %%\n');
%fprintf(figout,'    restrict y to domain=%d:%d, %%\n', S(3)/100, S(4)*100);
fprintf(figout,'	  axis background/.style = {fill=white,opacity = 1}]\n\n');

if (taylor)
	% draw \mu0
	fprintf(figout,'    \\draw[SPECred,thick] (axis cs:%e,%e) --',t0,S(3));
	fprintf(figout,'    (axis cs:%e,%e);\n\n',t0,S(4));
end

minkk = sum(xl<S(1))+1;
maxkk = sum(xl<=S(2));

minkk = min(minkk,indmin-2);
maxkk = max(maxkk,indmin);


for ii=1:p
	labels{ii} = num2str(ii-1);
	el = zeros(1,n); 
	el(ii) = 1;
	fprintf(figout,'    \\addplot\n');
	fprintf(figout,'    coordinates{%%\n');
	for kk = minkk:maxkk
		if (((zzz(kk,ii)<=S(4)) && (zzz(kk,ii)>=S(3))) || (abs(kk-indmin)<=1))
			fprintf(figout,'    (%e,%e)%%\n',xl(kk),zzz(kk,ii));
		end
	end			
	fprintf(figout,'    };\n\n');

end
fprintf(figout,'	\\end{semilogyaxis}%%\n');


fprintf(figout,'	\\draw (current axis.north west)--(plot4aul); %%\n');
fprintf(figout,'	\\draw (current axis.south east)--(plot4alr); %%\n');
end

fprintf(figout,'\\end{tikzpicture}%%\n');
fprintf(figout,'\\endpgfgraphicnamed%%\n');


if (type=='pap')
	if (usevpa)||(usesingle)
		if (taylor)
			fprintf(figout,'\\caption{Example~\\ref{example:%d}---Maximal difference between Taylor approximations and eigenvalues using single precision to store $E$, $\\mu_{0}=%3.2f$.}%%\n',example,t0);
		else
			fprintf(figout,'\\caption{Example~\\ref{example:%d}---Maximal difference between Chebyshev	approximations and eigenvalues using single precision to store $E$, $[\\mu_{1},\\mu_{2}] = [%3.2f,%3.2f]$.}%%\n',example,T2(1),T2(2));
		end
	else
		if (taylor)
			fprintf(figout,'\\caption{Example~\\ref{example:%d}---Maximal difference between Taylor approximations and eigenvalues, $\\mu_{0}=%3.2f$.}%%\n',example,t0);
		else
			fprintf(figout,'\\caption{Example~\\ref{example:%d}---Maximal difference between Chebyshev approximations and eigenvalues, $[\\mu_{1},\\mu_{2}] = [%3.2f,%3.2f]$.}%%\n',example,T2(1),T2(2));
		end
	end
	if (taylor)
		if (usevpa)||(usesingle)
			fprintf(figout,'\\label{fig:example4vpa_%d_%d_%d_%3.2f_%d}%%\n',taylor,n,p,t0,example);
		else
			fprintf(figout,'\\label{fig:example4_%d_%d_%d_%3.2f_%d}%%\n',taylor,n,p,t0,example);
		end		
	else	
		if (usevpa)||(usesingle)
			fprintf(figout,'\\label{fig:example4vpa_%d_%d_%d_%3.2f_%d}%%\n',taylor,n,p,T2(1),example);
		else
			fprintf(figout,'\\label{fig:example4_%d_%d_%d_%3.2f_%d}%%\n',taylor,n,p,T2(1),example);
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
