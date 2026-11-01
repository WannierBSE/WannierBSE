%==========================================================================
% WF_PLOT
%
% DESCRIPTION:
%   Plots Wannier-function isosurfaces for visual inspection of spinor real and imaginary components.
%
% INPUT ARGUMENTS:
%   None. Input files are loaded from the configured/current Wannier workspace.
%
% OUTPUT:
%   None. Isosurface plots are displayed and optionally saved.
%==========================================================================

clear
clc
close all

CurrentPath = pwd;
CalculationPath = fullfile(CurrentPath,'../../functions_forBSE/');


isovalue = 0.2;
tolerance = 0.02;
Save = 0;

%----------read lattice vectors---------
[a1,a2,a3,a,tau,N_atoms] = read_structure;                    % Lattice vectors and atom positions
%----------read lattice vectors---------

%--------------------------------------------------------------------------
cd(CurrentPath)
cd ..

load('W_up.mat')
load('W_down.mat')
load('r.mat')

TM_position = tau(1,:);
TopC_position = tau(2,:);
BottomC_position = tau(3,:);
%--------------------------------------------------------------------------


cd(CurrentPath)

Title_Fs = 20;
AtomMksize = 80;
mksize = 5;
azimuth = 40;
elevation = 15;


Xmin = TM_position(1)-3;
Xmax = TM_position(1)+3;
Ymin = TM_position(2)-3;
Ymax = TM_position(2)+3;
Zmin = TM_position(3)-4;
Zmax = TM_position(3)+4;

mks = 30;

         

for j=1:size(W_up,2)


figure(j)

%-------------------------up real -------------------------------
subplot(2,2,1)

isodata_plus = find(abs(real(W_up(:,j))-isovalue)<tolerance);
isodata_minus = find(abs(real(W_up(:,j))-(-isovalue))<tolerance);

scatter3(r(isodata_plus,1),r(isodata_plus,2),r(isodata_plus,3),mksize,'r','filled');hold on
scatter3(r(isodata_minus,1),r(isodata_minus,2),r(isodata_minus,3),mksize,'b','filled');hold on

plot3([TM_position(1),TopC_position(1)],[TM_position(2),TopC_position(2)],[TM_position(3),TopC_position(3)],'k-','linewidth',2);hold on
plot3([TM_position(1),BottomC_position(1)],[TM_position(2),BottomC_position(2)],[TM_position(3),BottomC_position(3)],'k-','linewidth',2);hold on
scatter3(TM_position(1),TM_position(2),TM_position(3),AtomMksize,[0.5,0.5,0.5],'filled','markeredgecolor','k','linewidth',1.5);hold on
scatter3(TopC_position(1),TopC_position(2),TopC_position(3),AtomMksize,'y','filled','markeredgecolor','k','linewidth',1.5);hold on
scatter3(BottomC_position(1),BottomC_position(2),BottomC_position(3),AtomMksize,'y','filled','markeredgecolor','k','linewidth',1.5);hold on

view(azimuth,elevation)

axis equal
xlim([Xmin,Xmax])
ylim([Ymin,Ymax])
zlim([Zmin,Zmax])

title([num2str(j),': up real'],'interpreter','latex','fontsize',15)
xlabel('$x(\AA)$','interpreter','latex','fontsize',15)
ylabel('$y(\AA)$','interpreter','latex','fontsize',15)
zlabel('$z(\AA)$','interpreter','latex','fontsize',15)
%-------------------------up real -------------------------------



%-------------------------up imag -------------------------------
subplot(2,2,2)

isodata_plus = find(abs(imag(W_up(:,j))-isovalue)<tolerance);
isodata_minus = find(abs(imag(W_up(:,j))-(-isovalue))<tolerance);

scatter3(r(isodata_plus,1),r(isodata_plus,2),r(isodata_plus,3),mksize,'r','filled');hold on
scatter3(r(isodata_minus,1),r(isodata_minus,2),r(isodata_minus,3),mksize,'b','filled');hold on

plot3([TM_position(1),TopC_position(1)],[TM_position(2),TopC_position(2)],[TM_position(3),TopC_position(3)],'k-','linewidth',2);hold on
plot3([TM_position(1),BottomC_position(1)],[TM_position(2),BottomC_position(2)],[TM_position(3),BottomC_position(3)],'k-','linewidth',2);hold on
scatter3(TM_position(1),TM_position(2),TM_position(3),AtomMksize,[0.5,0.5,0.5],'filled','markeredgecolor','k','linewidth',1.5);hold on
scatter3(TopC_position(1),TopC_position(2),TopC_position(3),AtomMksize,'y','filled','markeredgecolor','k','linewidth',1.5);hold on
scatter3(BottomC_position(1),BottomC_position(2),BottomC_position(3),AtomMksize,'y','filled','markeredgecolor','k','linewidth',1.5);hold on

view(azimuth,elevation)

axis equal
xlim([Xmin,Xmax])
ylim([Ymin,Ymax])
zlim([Zmin,Zmax])

title([num2str(j),': up imag'],'interpreter','latex','fontsize',15)
xlabel('$x(\AA)$','interpreter','latex','fontsize',15)
ylabel('$y(\AA)$','interpreter','latex','fontsize',15)
zlabel('$z(\AA)$','interpreter','latex','fontsize',15)
%-------------------------up imag -------------------------------




%-------------------------down real -------------------------------
subplot(2,2,3)

isodata_plus = find(abs(real(W_down(:,j))-isovalue)<tolerance);
isodata_minus = find(abs(real(W_down(:,j))-(-isovalue))<tolerance);

scatter3(r(isodata_plus,1),r(isodata_plus,2),r(isodata_plus,3),mksize,'r','filled');hold on
scatter3(r(isodata_minus,1),r(isodata_minus,2),r(isodata_minus,3),mksize,'b','filled');hold on

plot3([TM_position(1),TopC_position(1)],[TM_position(2),TopC_position(2)],[TM_position(3),TopC_position(3)],'k-','linewidth',2);hold on
plot3([TM_position(1),BottomC_position(1)],[TM_position(2),BottomC_position(2)],[TM_position(3),BottomC_position(3)],'k-','linewidth',2);hold on
scatter3(TM_position(1),TM_position(2),TM_position(3),AtomMksize,[0.5,0.5,0.5],'filled','markeredgecolor','k','linewidth',1.5);hold on
scatter3(TopC_position(1),TopC_position(2),TopC_position(3),AtomMksize,'y','filled','markeredgecolor','k','linewidth',1.5);hold on
scatter3(BottomC_position(1),BottomC_position(2),BottomC_position(3),AtomMksize,'y','filled','markeredgecolor','k','linewidth',1.5);hold on

view(azimuth,elevation)

axis equal
xlim([Xmin,Xmax])
ylim([Ymin,Ymax])
zlim([Zmin,Zmax])

title([num2str(j),': down real'],'interpreter','latex','fontsize',15)
xlabel('$x(\AA)$','interpreter','latex','fontsize',15)
ylabel('$y(\AA)$','interpreter','latex','fontsize',15)
zlabel('$z(\AA)$','interpreter','latex','fontsize',15)
%-------------------------down real -------------------------------



%-------------------------down imag -------------------------------
subplot(2,2,4)

isodata_plus = find(abs(imag(W_down(:,j))-isovalue)<tolerance);
isodata_minus = find(abs(imag(W_down(:,j))-(-isovalue))<tolerance);

scatter3(r(isodata_plus,1),r(isodata_plus,2),r(isodata_plus,3),mksize,'r','filled');hold on
scatter3(r(isodata_minus,1),r(isodata_minus,2),r(isodata_minus,3),mksize,'b','filled');hold on

plot3([TM_position(1),TopC_position(1)],[TM_position(2),TopC_position(2)],[TM_position(3),TopC_position(3)],'k-','linewidth',2);hold on
plot3([TM_position(1),BottomC_position(1)],[TM_position(2),BottomC_position(2)],[TM_position(3),BottomC_position(3)],'k-','linewidth',2);hold on
scatter3(TM_position(1),TM_position(2),TM_position(3),AtomMksize,[0.5,0.5,0.5],'filled','markeredgecolor','k','linewidth',1.5);hold on
scatter3(TopC_position(1),TopC_position(2),TopC_position(3),AtomMksize,'y','filled','markeredgecolor','k','linewidth',1.5);hold on
scatter3(BottomC_position(1),BottomC_position(2),BottomC_position(3),AtomMksize,'y','filled','markeredgecolor','k','linewidth',1.5);hold on

view(azimuth,elevation)

axis equal
xlim([Xmin,Xmax])
ylim([Ymin,Ymax])
zlim([Zmin,Zmax])

title([num2str(j),': down imag'],'interpreter','latex','fontsize',15)
xlabel('$x(\AA)$','interpreter','latex','fontsize',15)
ylabel('$y(\AA)$','interpreter','latex','fontsize',15)
zlabel('$z(\AA)$','interpreter','latex','fontsize',15)
%-------------------------down imag -------------------------------

f = gcf;
f.Position = [200,100,1000,800];


if Save==1
cd figures/
set(gcf,'InvertHardcopy','on');

print(gcf,['isosurface',num2str(j)],'-dpng','-r600');

cd ..
end

end



cd(CurrentPath)
