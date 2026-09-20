clear all

filenames = ["data/err_BDF1_HS1.mat","data/err_BDF2_HS.mat","data/err_RadauIIA2_HS.mat","data/err_RadauIIA3_HS.mat"];
numd = length(filenames);
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');

symbols='*oxsd+^.v><';
ms=[6 6 6 8 6 6 6 6 6 6 6];
gr=(linspace(.66,0,2))';

colors = ["#0072BD","#77AC30","#EDB120","#D95319","#7E2F8E"];
h1 = figure();
set(h1,'Position',[10 10 600 420])

for j=1:numd
    filenames(j)
    load(filenames(j))
  %  subplot(1,2,1)
      loglog(Ns,err,'LineWidth',1,'Marker',symbols(2),...
           'MarkerSize',ms(1),'Color',colors(j))
         hold on
    loglog(counters,errf,'LineWidth',1,'Marker',symbols(1),...
           'MarkerSize',ms(2),'Color',colors(j))

 %   subplot(1,2,2)



end
%loglog(Ns,50*Ns.^(-4/3),'--k')

xlim([10^1,10^5.2])
ylim([10^(-5.2),10^(-0.2)])
xlabel('Number of Laplace evaluations','Interpreter','Latex')
ylabel('Weighted $\ell^2-$ error','Interpreter','Latex')
legend(strcat('Implicit Euler'),strcat('Parsimonious Implicit Euler'),strcat('BDF-$2$'),strcat('Parsimonious BDF-$2$'),...
    strcat('RadauIIA-$2$'), strcat('Parsimonious RadauIIA-$2$'), ...
    strcat('RadauIIA-$3$'),strcat('Parsimonious RadauIIA-$3$'),...
     'location','southwest','Interpreter','Latex'); 

% legend(strcat('$|K(\partial_t^\tau) g - K(\partial_t) g |_2$'), ...
%     strcat('$|I_pK(\partial_t^\tau) g - K(\partial_t) g |_2$'),...
%      strcat('$\mathcal O\big(\frac{1}{\sqrt{L}}\big)$'), strcat('$\mathcal O\big(\frac{1}{L}\big)$') ...
%      ,'location','northeast','Interpreter','Latex'); 
hold off
title("Time discretization error")

saveas(gcf,'plots/comparison_HS','epsc') 