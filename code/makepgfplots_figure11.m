%% Plot of sampled eigenvalues and Taylor/Chebushev approximation
% this file produces a tex-file that is directly inserted in the paper

fprintf('Running makepgfplots_figure11.m; type: %s\n',type);

if (taylor)
	
	if (usevpa)
		filename = sprintf('exp_%s_ksdensity_taylor_%d_%d_%f_%d_vpa.tex',type,n,p,t0,example);
	else
		filename = sprintf('exp_%s_ksdensity_taylor_%d_%d_%f_%d.tex',type,n,p,t0,example);
	end

else
	
	if (usevpa)
		filename = sprintf('exp_%s_ksdensity_chebyshev_%d_%d_%f_%f_%d_vpa.tex',type,n,p,T2(1),T2(2),example);
	else
		filename = sprintf('exp_%s_ksdensity_chebyshev_%d_%d_%f_%f_%d.tex',type,n,p,T2(1),T2(2),example);
	end
	
end
fprintf('Writing file %s\n',filename);
figout = fopen(filename,'w');


if (~taylor)

	fp{1} = @(mu) dp(:,1)*Usch{1}(mu);
	fvp{1} = @(mu) vp(:,:,1)*Usch{1}(mu);
	fnp{1} = @(mu) inv(fvp{1}(mu))*x;
	dfpdmu{1} = null; % derivative of fp 


	for kk = 2:p
		fp{kk} = @(mu) fp{kk-1}(mu) + dp(:,kk)*Usch{kk}(mu);
		fvp{kk} = @(mu) fvp{kk-1}(mu) + vp(:,:,kk)*Usch{kk}(mu);
		fnp{kk} = @(mu) inv(fvp{kk}(mu))*x;
		dfpdmu{kk} = @(mu) dfpdmu{kk-1}(mu) + dp(:,kk)*dUsch{kk}(mu);
	end

end


% pick some mus
if (~exist('mus','var'))
	mus = linspace(2, 4, 15); % starting guess for mu
end	

% store guesses
t = [];
w = [];
wd = [];
smu = [];

for kk = 1:length(mus)
 	ev = fp{p}(mus(kk))';
 	dev = dfpdmu{p}(mus(kk))';
	g = fnp{p}(mus(kk))';
	[m,n]=size(g);
	
	dmu = (g(2:timesteps*realizations,:)./g(1:(timesteps*realizations-1),:));
	for ll = 1:realizations
		for ii = 1:timesteps-1
			%for ii = 1:max(s)-1
			emu((ll-1)*(timesteps-1)+ii,:) = (dmu((ll-1)*timesteps+ii,:) - exp(h*ev))./(exp(h*ev).*(h*dev));
		end	
	end

	for ii = 1:n
		t = [t; mus(kk) + emu(1:(timesteps-1)*realizations,ii)];
		smu = [smu; mus(kk)];
		w = [w; 1./(0.1 + emu(1:(timesteps-1)*realizations,ii))];
		wd = [wd; abs(emu(1:(timesteps-1)*realizations,ii))];
	end
end

ts = t(t<T2(2) & t>T2(1));
ws = w(t<T2(2) & t>T2(1));
%[f2,xi2,bw2] = ksdensity(t(t<T2(2) & t>T2(1)),'Support',T2,'NumPoints',10001);	
if (isclose)
	[f,xi,bw] = mvksdensity(ts,'NumPoints',10001);
	%[f,xi,bw] = mvksdensity(ts,'Support',T2,'NumPoints',10001,'Weights',ws);	
else	
	%[f,xi,bw] = mvksdensity(ts,'Support',T2,'NumPoints',10001);
	[f,xi,bw] = ksdensity(ts,'NumPoints',10001);
end
wds = wd(t<T2(2) & t>T2(1))*bw;

ff = zeros(size(f));
for kk=1:length(f)
	ff(kk) = sum(abs(1./wds).*exp(-(((xi(kk)-ts)./wds).^2)/2)/sqrt(2*pi));
end
ff2 = ff/(sum(ff*(xi(2)-xi(1))));

%norm(f-f2)
%norm(xi-xi2)
%norm(bw-bw2)
figure
plot(xi,f)
hold on
%plot(xi,ff2)
af = argmax(f)
best = xi(af)
plot([best, best],[0,f(af)])



 	
if (type=='pap')
	fprintf(figout,'\\begin{figure}\n');
 	fprintf(figout,'\\centering\n');
end
fprintf(figout,'\\beginpgfgraphicnamed{figure-%d-%d-%d}\n',n,p,round(t0*100));
fprintf(figout,'\\begin{tikzpicture}\n');
fprintf(figout,'  \\begin{axis}[ %%\n');
fprintf(figout,'    remember picture, %%\n');
fprintf(figout,'    name = plot1a, %%\n');
fprintf(figout,'    set layers=axis on top,%%\n');
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
fprintf(figout,'    ymin = %e, ymax = %e, %% change if needed\n', 0, 2);
fprintf(figout,'    restrict x to domain=%d:%d, %%\n', floor(T(1))-2, ceil(T(2))+2);
fprintf(figout,'    xlabel = {$\\mu$},%%\n');
fprintf(figout,'    ylabel = {Density},%%\n');
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
 
fprintf(figout,'    \\addplot[SPECblue, thick]\n');
fprintf(figout,'    coordinates{%%\n');
for kk = 1:length(xi)
	fprintf(figout,'    (%e,%e)%%\n',xi(kk),f(kk));
end
fprintf(figout,'    };%%\n');
fprintf(figout,'    \\draw (axis cs:%e,%e) -- ',best,0);
fprintf(figout,'(axis cs:%e,%e);%%\n',best,f(af));
fprintf(figout,'  \\end{axis}%%\n');
fprintf(figout,'\\end{tikzpicture}%%\n');
fprintf(figout,'\\endpgfgraphicnamed%%\n');
if (taylor)
	fprintf(figout,'\\caption{Example~\\ref{example:%d}, $n=%d$, $\\mu_{0}=%3.2f$.}%%\n',example,n,t0);
else
	fprintf(figout,'\\caption{Example~\\ref{example:%d}, $n=%d$, $[\\mu_{1},\\mu_{2}]=[%3.2f,%3.2f]$.}%%\n',example,n,T2(1),T2(2));
end

if (taylor)
	if (usevpa)||(usesingle)
		fprintf(figout,'\\label{fig11:example1vpa_%d_%d_%d_%3.2f_%d}%%\n',taylor,n,p,t0,example);
	else
		fprintf(figout,'\\label{fig11:example1_%d_%d_%d_%3.2f_%d}%%\n',taylor,n,p,t0,example);
	end		
else	
	if (usevpa)||(usesingle)
		fprintf(figout,'\\label{fig11:example1vpa_%d_%d_%d_%3.2f_%d}%%\n',taylor,n,p,T2(1),example);
	else
		fprintf(figout,'\\label{fig11:example1_%d_%d_%d_%3.2f_%d}%%\n',taylor,n,p,T2(1),example);
	end		
end

fprintf(figout,'\\end{figure}\n\n\n');
fprintf(figout,'%%%% Thomas'' emacs local variable set up\n\n');
fprintf(figout,'%%%%%% Local Variables:\n'); 
fprintf(figout,'%%%%%% mode: LaTeX\n');
if (type=='tal')
 	fprintf(figout,'%%%%%% TeX-master: "parametric_evp"\n');
else
 	fprintf(figout,'%%%%%% TeX-master: "paper1"\n');
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
 	
