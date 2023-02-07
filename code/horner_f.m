function [x] = horner_f(mu, t0, coeff)

% Returns value of 
% coeff(1) + coeff(2) (mu-t0) + coeff(3) * (mu-t0)^2 + ...
% based on Horner's method
% if coeff is a vector/matrix the corresponding matrix polynomial is computed
% coeff is a m x n x d object
% with d-1 the degree of the polynomial

dims = size(coeff);
dim = length(dims);

x = zeros([dims(1:end-1),length(mu)]);
d = dims(dim) - 1;

if (dim==3)
	for jj = 1:length(mu)	
		
		%x(:,:,jj) = coeff(:,:,d+1)/factorial(d);
		x(:,:,jj) = coeff(:,:,d+1);
		for kk = d:-1:1
			%x(:,:,jj) = x(:,:,jj)*(mu(jj)-t0) + coeff(:,:,kk)/factorial(kk-1);
			x(:,:,jj) = x(:,:,jj)*(mu(jj)-t0) + coeff(:,:,kk);
		end
		
	end
elseif (dim==2)
	
	for jj = 1:length(mu)	
		
		%x(:,jj) = coeff(:,d+1)/factorial(d);
		x(:,jj) = coeff(:,d+1);
		for kk = d:-1:1
			%x(:,jj) = x(:,jj)*(mu(jj)-t0) + coeff(:,kk)/factorial(kk-1);
			x(:,jj) = x(:,jj)*(mu(jj)-t0) + coeff(:,kk);
		end
		
	end
end





%%% Local Variables: 
%%% mode:matlab
%%% flyspell-mode:nil
%%% mode:flyspell-prog
