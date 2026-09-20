clear all

%load('../data/errors_regular_fixdelta.mat')
load('../data/errorsh05cube.mat')
%load('../data/errf.mat')
%load('../data/errors_gaussian_large.mat')
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');


symbols='*oxsd+^.v><';
ms=[6 6 6 8 6 6 6 6 6 6 6];
gr=(linspace(.66,0,2))';
colors=[gr gr gr];

h1 = figure();
set(h1,'Position',[10 10 1000 420])
subplot(1,2,1)
loglog(Ns,err,'LineWidth',1,'Marker',symbols(1),...
           'MarkerSize',ms(1))
hold on
loglog(counters,errf,'LineWidth',1,'Marker',symbols(2),...
           'MarkerSize',ms(2))

loglog(Ns,0.5*Ns.^(-1),'--k')
loglog(Ns,20*Ns.^(-2),'--k')
%ylim([10^(-4),10^(-1.5)])
xlabel('Amount of Laplace evaluations','Interpreter','Latex')
ylabel('Error')
legend(strcat('BDF-$1$'), ...
    strcat('modified BDF-$1$'),...
     strcat('$\mathcal O\big(\frac{1}{L}\big)$'), strcat('$\mathcal O\big(\frac{1}{L^2}\big)$') ...
     ,'location','northeast'); 
% legend(strcat('$|K(\partial_t^\tau) g - K(\partial_t) g |_2$'), ...
%     strcat('$|I_pK(\partial_t^\tau) g - K(\partial_t) g |_2$'),...
%      strcat('$\mathcal O\big(\frac{1}{L}\big)$'), strcat('$\mathcal O\big(\frac{1}{L^2}\big)$') ...
%      ,'location','northeast'); 
hold off
title("Time discretization error")
subplot(1,2,2)

loglog(Ns,errfvsn,'LineWidth',1,'Marker',symbols(1),...
           'MarkerSize',ms(1))
xlabel('Amount of timesteps $N$','Interpreter','Latex')
ylabel('Error')
title("Error caused by interpolation")
hold on
%ylim([10^(-4),10^(-1.5)])
loglog(Ns,0.5*Ns.^(-1),'--k')
legend(strcat('$|K(\partial_t^\tau) g - I^{\Phi}_pK(\partial_t^\tau) g |_2$'), ...
        strcat('$\mathcal O\big(\frac{1}{N}\big)$'),...
     'location','northeast'); 
%saveas(gcf,'plots/ac_scat','epsc')  
% ERR2=ERR;
% load('../data/Err_data_delta01.mat')
%load('data/Err_data_newsigns2.mat')
% ERR = errors;
% tau_s = 1*cast(N_s,"double").^(-1);
% ERR(:,1:length(ERR(1,:)))
% 
% double_error_plots(errors,errors_est,eps_s,tau_s, "NN", 3,1)
