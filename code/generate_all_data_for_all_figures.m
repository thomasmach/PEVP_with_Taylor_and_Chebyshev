% run this file to generate the data for all figures

for select = [1,2,3,4,57,6,8,9,11,12,13,14,15,16]
	clearvars -except select
	experiments_paper
end

for select = [1,2]
	clearvars -except select
	experiments_sampling_2
end

runtime_test

%%% Local Variables: 
%%% mode:matlab
%%% flyspell-mode:nil
%%% mode:flyspell-prog
