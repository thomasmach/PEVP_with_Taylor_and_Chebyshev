% Generate Chebyshev polynomials of second kind using Chebfun
for ii = 1:p
	Usch{ii} = sqrt(2)*chebpoly(ii-1,T2,2)/sqrt(pi());
end
	
