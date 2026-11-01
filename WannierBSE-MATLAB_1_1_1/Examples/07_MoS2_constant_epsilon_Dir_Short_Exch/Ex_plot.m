% =========================================================================
% PLOT_EXCITON_LEVELS.M
% =========================================================================
% Summary:   Loads and visualizes exciton energy spectra from Ex.mat data.
%
% Features:
%   - Aspect Ratio Control for publication-quality visuals.
%   - Automated multi-format export (PDF/PNG).
% =========================================================================

clear; clc; close all;

%% ------------------------------------------------------------------------
%  1. GLOBAL USER CONFIGURATION (EDIT HERE)
%  ------------------------------------------------------------------------

% -- File & Path Settings --
config.dataPath     = 'Exciton_data';  % Folder containing the results
config.dataFile     = 'Ex.mat';         % Input filename
config.pngOutput    = 'Ex_plot.png';    % Export filename (Raster)
%config.pdfOutput   = 'Ex_plot.pdf';    % Uncomment to export PDF (Vector)

% -- Visual Styling --
config.lineWidth    = 2.0;              % Thickness of energy level lines
config.lineColor    = [0.1, 0.25, 0.60];        % RGB Color (Blue)
config.fontSize     = 18;               % Base font size for ticks
config.fontName     = 'Helvetica';      % Professional font face
config.aspectRatio  = 2.8;              % Y/X ratio for the plot box

% -- Export Parameters --
config.dpi          = '-r300';          % PNG Resolution (300 DPI)
config.rightMargin  = 0.30;             % Buffer space for LaTeX labels (inches)

%% ------------------------------------------------------------------------
%  2. DATA INITIALIZATION & LOADING
%  ------------------------------------------------------------------------

fprintf('\n>> Initializing Exciton Plotting Suite...\n');

scriptDir = fileparts(mfilename('fullpath'));
filePath = fullfile(scriptDir, config.dataPath, config.dataFile);

% Check for file existence - Abort if not found
if ~exist(filePath, 'file')
    error('Error: Target file [%s] not found.', filePath);
end

fprintf('>> Loading dataset: %s\n', filePath);
load(filePath, 'Ex'); 

%% ------------------------------------------------------------------------
%  3. CORE PLOTTING ENGINE
%  ------------------------------------------------------------------------

% Initialize Figure
fig = figure('Color', 'white', 'Name', 'Exciton Energy Spectrum');
set(gcf, 'Position', [100, 100, 400, 600]); 
ax = axes('Parent', fig);

% Render Energy Levels
hold(ax, 'on');
for i = 1:length(Ex)
    line(ax, [0, 1], [Ex(i), Ex(i)], ...
        'Color', config.lineColor, ...
        'LineWidth', config.lineWidth);
end

% Set Axis Proportions
pbaspect(ax, [1, config.aspectRatio, 1]);

% Styling & Limits
xlim(ax, [0, 1]);
set(ax, 'XTick', []); 

% LaTeX Labeling
ylabel(ax, '$\mathrm{Exciton\ energy}\ E^{X} \ \mathrm{(eV)}$', ...
    'Interpreter', 'latex', ...
    'FontSize', config.fontSize + 2); 

% Calculate Padding
ylimits = ylim(ax);
ylim_padding = (ylimits(2) - ylimits(1)) * 0.05; 
ylim(ax, [ylimits(1) - ylim_padding, max(Ex)]);

% Final Axis Formatting
set(ax, ...
    'FontName',     config.fontName, ...
    'FontSize',     config.fontSize, ...
    'Box',          'on', ...
    'LineWidth',    1.5, ...
    'TickDir',      'in', ...
    'TickLength',   [0.02 0.02], ...
    'YMinorTick',   'off', ... 
    'Layer',        'top');

%% ------------------------------------------------------------------------
%  4. EXPORT & FILESYSTEM MANAGEMENT
%  ------------------------------------------------------------------------

% Prepare output directory and file paths
outputRelativePath = fullfile(config.dataPath, 'Figures');
outputDir = fullfile(scriptDir, outputRelativePath);
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

pngPath = fullfile(outputDir, config.pngOutput);
if isfield(config, 'pdfOutput') && ~isempty(config.pdfOutput)
    pdfPath = fullfile(outputDir, config.pdfOutput);
end

% Tight Padding Logic
set(ax, 'Units', 'normalized');
set(fig, 'PaperPositionMode', 'auto'); 
set(fig, 'PaperUnits', 'inches');

fig_pos = get(fig, 'PaperPosition');
set(fig, 'PaperPosition', [0 0 fig_pos(3) fig_pos(4)]);

% Set Paper Size with margins
paperWidth  = fig_pos(3) + config.rightMargin;
paperHeight = fig_pos(4) + 0.01;
set(fig, 'PaperSize', [paperWidth, paperHeight]); 

% Output to Disk
fprintf('>> Exporting publication-quality files...\n');
print(fig, pngPath, '-dpng', config.dpi); 
if isfield(config, 'pdfOutput') && ~isempty(config.pdfOutput)
    print(fig, pdfPath, '-dpdf', '-painters');
end

fprintf('[Info] The Exciton energy spectrum was generated at %s\n', ...
    outputRelativePath);
% =========================================================================
% END OF SCRIPT
% =========================================================================