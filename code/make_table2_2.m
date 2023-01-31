if (usevpa)
	filename = sprintf('exp_pap_table41_%d_%d_%f_%d_vpa.tex',n,p,t0,example);
else
	filename = sprintf('exp_pap_table41_%d_%d_%f_%d.tex',n,p,t0,example);
end

figout = fopen(filename,'w');

fprintf(figout,'\\scalebox{0.8}{\n');
fprintf(figout,'\\begin{tabular}{l');

for ii = 1:p
	fprintf(figout,'r');
end

fprintf(figout,'}\n\\toprule\n');
fprintf(figout,'ev ');

for ii = 1:p
	fprintf(figout,'& $\\lambda_{%d}$',ii-1);
end

fprintf(figout,'\\\\\n\\midrule\n');

for ii = 1:n
	fprintf(figout,'%d ',ii);

	for jj = 1:p
		h = dp(ii,jj);
		exponent = floor(log10(abs(h)));
		f = h*10^-exponent;
		
		fprintf(figout,'& %3.2f$_{10^{%+03d}}$',f,exponent);
	end
	
	fprintf(figout,'\\\\\n');

end

fprintf(figout,'\\midrule\n');
fprintf(figout,'$\\sum$ ',ii);

for jj = 1:p
	h = sum(dp(:,jj));
	exponent = floor(log10(abs(h)));
	f = h*10^-exponent;
	
	fprintf(figout,'& %3.2f$_{10^{%+2d}}$',f,exponent);
end

fprintf(figout,'\\\\\n\\bottomrule\n');
fprintf(figout,'\\end{tabular}\n');
fprintf(figout,'}\n');

fclose(figout);
