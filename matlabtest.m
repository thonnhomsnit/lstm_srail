clc;
norm_x = [-1 -1 -1 -1; -1 -1 -1 1; -1 -1 -1 -1]; % mock input from NSGAII function
num = height(norm_x);
for i = 1:num
    disp(i+1);
    input = [0 0 0 0; norm_x(i,:)]; % send this input to lstm function
    EA = lstm(input);
    EAcollect(i)=EA;
end

function [EA] = lstm(input)

csvwrite('matlabinput.csv', input);

% run the executable file
system('C:\Users\Personal\Documents\GitHub\lstm_srail\dist\testmatlab\testmatlab.exe');

answer = csvread('pythonoutput.csv');
% prepare for Area calculation
force = answer(:,1);
disp = answer(:,2);

disp_offset = [0; disp];
force_offset = [0; force];
delta_disp = [disp; 0]-disp_offset;
sum_force = [force; 0]+force_offset;

% find last index that have disp < -0.7 mm
[row] = find(disp<-0.7, 1, 'last');
ea = 0.5.*delta_disp.*sum_force;
EA = 0;
for j = 1:row
        EA=EA+abs(ea(j)); % abs is used for temporary normalized data
end

end
% %% 
% %% check pythonoutput.csv result
% plot(test(:,2),test(:,1),'r-')
% hold on;
% plot(answer(:,2),answer(:,1),'b-')
% %% check output for NSGAII function
% x = [0 0 0 0; 0 0 0 1]; % mock in put (2 rows of normalized TR A L T)
% [Y,Xf,Af] = poslin81fn(x);  % output (2 rows of true IPF -SEA)
function [mass] = ANN_mass(input)

end

function [ipf] = ANN_ipf(input)

end
