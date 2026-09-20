clear all

%load('../data/errors_regular_fixdelta.mat')
load('data/err_BDF2_HS.mat')
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

loglog(Ns,errTR,'LineWidth',1,'Marker',symbols(2),...
           'MarkerSize',ms(2))
loglog(Ns,4*Ns.^(-2/3),'--k')
loglog(Ns,50*Ns.^(-4/3),'--k')
ylim([10^(-3),10^(0)])
xlabel('Number of Laplace evaluations','Interpreter','Latex')
ylabel('Weighted $\ell^2-$ error','Interpreter','Latex')
legend(strcat('BDF-2'), ...
    strcat('Parsimonious BDF-2'),strcat('Trapezoidal rule'),...
    strcat('$\mathcal O\big(L^{-2/3}\big)$'), strcat('$\mathcal O\big(L^{-4/3}\big)$') ...
     ,'location','northeast','Interpreter','Latex'); 
% legend(strcat('$|K(\partial_t^\tau) g - K(\partial_t) g |_2$'), ...
%     strcat('$|I_pK(\partial_t^\tau) g - K(\partial_t) g |_2$'),...
%      strcat('$\mathcal O\big(\frac{1}{\sqrt{L}}\big)$'), strcat('$\mathcal O\big(\frac{1}{L}\big)$') ...
%      ,'location','northeast','Interpreter','Latex'); 
hold off
title("Time discretization error")
subplot(1,2,2)

loglog(Ns,errfvsn,'LineWidth',1,'Marker',symbols(1),...
           'MarkerSize',ms(1))
xlabel('Number of timesteps $N$','Interpreter','Latex')
ylabel('Weighted $\ell^2-$ error','Interpreter','Latex')
title("Error caused by interpolation")
hold on
ylim([10^(-4),10^(0)])
loglog(Ns,10*Ns.^(-2/3),'--k')

 legend(strcat('$|K(\partial_t^\tau) g - I^{\mathcal{T}}_pK(\partial_t^\tau) g |_2$'), ...
         strcat('$\mathcal O\big(N^{-2/3}\big)$'),...
      'location','northeast'); 
saveas(gcf,'plots/BDF2_HS','epsc')  
% ERR2=ERR;
% load('../data/Err_data_delta01.mat')
%load('data/Err_data_newsigns2.mat')
% ERR = errors;
% tau_s = 1*cast(N_s,"double").^(-1);
% ERR(:,1:length(ERR(1,:)))
% 
% double_error_plots(errors,errors_est,eps_s,tau_s, "NN", 3,1)
