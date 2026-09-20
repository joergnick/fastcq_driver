function [] = double_error_plots( ERRS,ERRS_est,mesh_size,step_size,string1,s,figurenumber)
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');
%% Creating time convergence plots
h1=figure(figurenumber);
subplot(1,2,1)
set(h1,'Position',[10 10 1000 420])


 %   style_list={'-d','-o','-^','-h','-*','-+','-x','p','.','r','b','y'};


start_time=1;
end_time=6;
%% Creating the plots
start_space=1;
end_space=5;
vect=step_size(start_time:end_time);
err=ERRS(start_space:end_space,start_time:end_time);

[n1 n2]=size(err);
symbols='sox*d+^.v><';
ms=[6 6 6 8 6 6 6 6 6 6 6];
gr=(linspace(.66,0,n1))';
colors=[gr gr gr];
for jj=1:n1
    loglog(vect,err(jj,:), ...
           'LineWidth',1,...
           'Marker',symbols(jj),...
           'MarkerSize',ms(jj),...
           'Color', colors(jj,:));%'Color', 'black'); %
    if jj==1
        hold on;
    end
end

%% Plotting the reference line
loglog((step_size),step_size,'--k')
 %% Title
% strL2=strcat(string1,' s= ',num2str(s),' L2 time conv.');
%strL2=strcat(string1,' L^2-norm time conv.');
%title(strL2)
title("Time convergence error",'Interpreter','Latex')
%% Creating the legend
legend(strcat('$\varepsilon=10^{-1}$'),...
    strcat('$\varepsilon=10^{-2}$'),strcat('$\varepsilon=10^{-3}$'),...
    strcat('$\varepsilon=10^{-4}$'),strcat('$\varepsilon=10^{-5}$'),...
     strcat('$\mathcal O(h)$') ,'location','southeast'); 
%% Labelling the axis
xlabel('step size $h$','Interpreter','Latex')
ylabel('$L^2$- error','Interpreter','Latex')

xlim([10^(-3) 10^(-0.5)])  
 ylim([10^(-5) 10])
  hold off

%% Error estimators 
subplot(1,2,2)

start_time=1;
end_time=6;
%% Creating the plots
start_space=1;
end_space=5;
vect=step_size(start_time:end_time);
err=ERRS_est(start_space:end_space,start_time:end_time);

[n1 n2]=size(err);
symbols='sox*d+^.v><';
ms=[6 6 6 8 6 6 6 6 6 6 6];
gr=(linspace(.66,0,n1))';
colors=[gr gr gr];
for jj=1:n1
    loglog(vect,err(jj,:), ...
           'LineWidth',1,...
           'Marker',symbols(jj),...
           'MarkerSize',ms(jj),...
           'Color', colors(jj,:));%'Color', 'black'); %
    if jj==1
        hold on;
    end
end
loglog((step_size),step_size,'--k')

 %% Title
% strL2=strcat(string1,' s= ',num2str(s),' L2 time conv.');
%strL2=strcat(string1,' L^2-norm time conv.');
%title(strL2)
title("Projection error bound $h\sum_{j=0}^N \delta_j$",'Interpreter','Latex')
%% Creating the legend
% legend(strcat('$\varepsilon^2=10^{-5}$'),...
%     strcat('$\varepsilon^2=10^{-6}$'),strcat('$\varepsilon^2=10^{-7}$'),...
%     strcat('$\varepsilon^2=10^{-8}$'),strcat('$\varepsilon^2=10^{-9}$'),...
%     strcat('$\varepsilon^2=10^{-10}$') ,'location','southeast');  
legend(strcat('$\varepsilon=10^{-1}$'),...
    strcat('$\varepsilon=10^{-2}$'),strcat('$\varepsilon=10^{-3}$'),...
    strcat('$\varepsilon=10^{-4}$'),strcat('$\varepsilon=10^{-5}$'),...
     strcat('$\mathcal O(h)$') ,'location','southeast'); 
%% Labelling the axis
xlabel('step size $h$','Interpreter','Latex')
ylabel('Projection error bound $\overline{\delta}$','Interpreter','Latex')

xlim([10^(-3) 10^(-0.5)])  
 ylim([10^(-5) 10])
  hold off
% saveas(gcf,'timeconv','epsc')  
saveas(gcf,'timeconv_gaussian_impl','epsc')  
%% EPS CONVERGENCE PLOT 1
%% Comment in from here for space convergence plot
h2=figure(figurenumber+1);
subplot(1,2,1)
set(h2,'Position',[10 10 1000 420])
%set(h2,'Position',[10 10 900 420])
%subplot(1,2,1)


 %   style_list={'-d','-o','-^','-h','-*','-+','-x','p','.','r','b','y'};


start_time=1;
end_time=6;
%% Creating the plots
start_space=1;
end_space=5;

%vect=step_size(start_time:end_time);
mesh_size = sqrt(mesh_size);
vect = mesh_size(start_space:end_space);
err=ERRS(start_space:end_space,start_time:end_time)';
[n1 n2]=size(err);

symbols='sox*d+^.v><';
ms=[6 6 6 8 6 6 6 6 6 6 6];
gr=(linspace(.66,0,n1))';
colors=[gr gr gr];



for jj=1:n1
    loglog(vect,err(jj,:), ...
           'LineWidth',1,...
           'Marker',symbols(jj),...
           'MarkerSize',ms(jj),...
           'Color', colors(jj,:));%'Color', 'black'); %
    if jj==1
        hold on;
    end
end
%% Plotting the reference line

 loglog(mesh_size,10*mesh_size,'--k')
%  loglog(mesh_size,0.7*mesh_size.^1.5,'--k')
 %% Title
title("$\varepsilon-$ convergence of the error", 'Interpreter','Latex')
    hold off
xlim([10^(-5.5) 10^(-0.5)])  
ylim([10^(-5) 10])
    
%% Creating the legend
 l=legend({strcat('$N = 10$'),strcat('$N = 10\cdot2$'),...
     strcat('$N = 10\cdot2^2$'),strcat('$N = 10\cdot2^3$'),...
    strcat('$N = 10\cdot2^4$'),  '$\mathcal O(\varepsilon)$'},'Interpreter','Latex','location','southeast');
%% Labelling the axis
xlabel('Regularization parameter $\varepsilon$','Interpreter','Latex')
ylabel('$L^2$- error','Interpreter','Latex')
%ylabel('H^1-norm error')
hold off


%% EPS CONVERGENCE PLOT 2
%% Comment in from here for space convergence plot
subplot(1,2,2)
set(h2,'Position',[10 10 1000 420])
%set(h2,'Position',[10 10 900 420])
%subplot(1,2,1)


 %   style_list={'-d','-o','-^','-h','-*','-+','-x','p','.','r','b','y'};


start_time=1;
end_time=6;
%% Creating the plots
start_space=1;
end_space=5;

%vect=step_size(start_time:end_time);

vect = mesh_size(start_space:end_space);
err=ERRS_est(start_space:end_space,start_time:end_time)';
[n1 n2]=size(err);

symbols='sox*d+^.v><';
ms=[6 6 6 8 6 6 6 6 6 6 6];
gr=(linspace(.66,0,n1))';
colors=[gr gr gr];



for jj=1:n1
    loglog(vect,err(jj,:), ...
           'LineWidth',1,...
           'Marker',symbols(jj),...
           'MarkerSize',ms(jj),...
           'Color', colors(jj,:));%'Color', 'black'); %
    if jj==1
        hold on;
    end
end
%% Plotting the reference line

 loglog(mesh_size,40*mesh_size,'--k')
%  loglog(mesh_size,0.7*mesh_size.^1.5,'--k')
 %% Title
title("Projection error bound $h\sum_{j=0}^N \delta_j$", 'Interpreter','Latex')
    hold off
xlim([10^(-5.5) 10^(-.5)])  
ylim([10^(-5) 10])
    
%% Creating the legend
 l=legend({strcat('$N = 10$'),strcat('$N = 10\cdot2$'),...
     strcat('$N = 10\cdot2^2$'),strcat('$N = 10\cdot2^3$'),...
    strcat('$N = 10\cdot2^4$'), strcat('$N = 10\cdot2^5$'), strcat('$N = 10\cdot2^6$'),...
    '$\mathcal O(\varepsilon)$'},'Interpreter','Latex','location','southeast');
%% Labelling the axis
xlabel('Regularization parameter $\varepsilon$','Interpreter','Latex')
ylabel('$L^2$ projection error bound $\overline{\delta}$','Interpreter','Latex')
%ylabel('H^1-norm error')
hold off
% saveas(gcf,'epsconv','epsc')  
saveas(gcf,'epsconv_gaussian_impl','epsc')  
end

