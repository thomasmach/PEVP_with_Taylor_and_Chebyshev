function [z] = onestep(A,z,n,kk)


np1 = n+1;
nn = np1*kk;


for ni = 1:n
	
	F = zeros(nn,nn);
	w = -eye(nn,1);
	
	in = 2:np1;

	for ii = 1:kk
		
		for ln = ii:-1:1
			
			lind0 = np1*(ii-ln)+1;
			lind1 = np1*(ii-ln)+in;
			
			ilind0 = np1*(ln-1)+1;
			ilind1 = np1*(ln-1)+in;
			
			
			
			% updating already filled cells due to U_m U_n = U_m-n U_m-n+2 ... U_m+n
			
			for jjjn = ii-2*min(ii-ln,ln-1)-1:2:ii-1
				
				ind0 = np1*jjjn+1;
				ind1 = np1*jjjn+in;
				

				% -A_{ii-ll}'s lower triangular
				w(ind1) = w(ind1)-A(:,:,ln)*z(lind1,ni);
				
				F(ind1,lind1) = F(ind1,lind1)+z(ilind0,ni)*eye(n,n)-A(:,:,ln);
				
				
				% x_{ii-ll}^T on upper subdiagonal 
				w(ind0) = w(ind0) + ctranspose(z(ilind1,ni))*z(lind1,ni);
				
				if (ii==2*ln-1)
					F(ind0,lind1) = F(ind0,lind1) + 2*ctranspose(z(ilind1,ni));
				else
					F(ind0,lind1) = F(ind0,lind1) + ctranspose(z(ilind1,ni));
				end
				
				% first column
				w(ind1) = w(ind1) + z(ilind1,ni)*z(lind0,ni);
				
				F(ind1,lind0) = F(ind1,lind0) + z(ilind1,ni);
			end
			
		end
		
	end

	% Ehrlich-Aberth update of F
	%	for nj = 1:n
	%		if (nj ~= ni)
	%			F = F - w*transpose(z(:,ni)-z(:,nj))/(ctranspose(z(:,ni)-z(:,nj))*(z(:,ni)-z(:,nj)));
	%		end	
	%	end
	z(:,ni) = z(:,ni) - F\w;

	%keyboard
	
end