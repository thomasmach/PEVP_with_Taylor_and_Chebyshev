r = rand(1,60)*sum(f);
ind = ones(1,60);
for kk = 1:length(f);
	ind = ind + (r>f(kk));
 	r = r - f(kk); 
	%ind
end
mus2 = xi(ind);
