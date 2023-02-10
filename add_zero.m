step1 = (371:400:32371);
step2 = (400:400:32400);
zero50 = zeros(30,1);
data0=data;
for i = 1:81
    data0(step1(i):step2(i),5) = zero50;
end
plot(data0(:,5))
%%
for i = 1:81
    plot(disp(:,i),'-');
    hold on;
end
xlabel('timestep');
ylabel('Displacement \rm (mm)')