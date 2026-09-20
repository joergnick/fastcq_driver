clear all


load('data/err_RadauIIA2_HS.mat')

set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');


symbols='*oxsd+^.v><';
ms=[6 6 6 8 6 6 6 6 6 6 6];
gr=(linspace(.66,0,2))';
colors=[gr gr gr];

h1 = figure();
subplot(1,2,1)
set(h1,'Position',[10 10 1000 420])

loglog(Ns,err,'LineWidth',1,'Marker',symbols(1),...
           'MarkerSize',ms(1))
hold on
loglog(counters,errf,'LineWidth',1,'Marker',symbols(2),...
           'MarkerSize',ms(2))

loglog(Ns,3*Ns.^(-3/4),'--k')
%loglog(Ns,50*Ns.^(-4/3),'--k')
ylim([10^(-4),10^(-1)])
xlabel('Number of Laplace evaluations','Interpreter','Latex')
ylabel('Weighted $\ell^2-$ error','Interpreter','Latex')
legend(strcat('RadauIIA-$2$'), ...
    strcat('Parsimonious RadauIIA-$3$'),...
     strcat('$\mathcal O\big(L^{-3/4}\big)$')...
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
ylim([10^(-12),10^(-4)])
%loglog(Ns,10*Ns.^(-4/3),'--k')

 legend(strcat('$|K(\partial_t^\tau) g - I^{\mathcal{T}}_pK(\partial_t^\tau) g |_2$') ,...
      'location','northeast'); 
 hold off
saveas(gcf,'plots/RadauIIA_2_HS','epsc') 