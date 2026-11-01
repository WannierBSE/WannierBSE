function [B,ndx,dbg] = natsortfiles(A,rgx,varargin)
%==========================================================================
% NATSORTFILES
%
% DESCRIPTION:
%   Sorts file or folder names using natural-order numeric token comparison while respecting path separators.
%
% INPUT ARGUMENTS:
%   A : file or folder names, paths, or directory structures.
%   rgx : optional tokenization regular expression.
%   varargin : optional sorting options.
%
% OUTPUT:
%   B : naturally sorted paths.
%   ndx : sort indices.
%   dbg : tokenization diagnostics.
%==========================================================================

fnh = @(c)cellfun('isclass',c,'char') & cellfun('size',c,1)<2 & cellfun('ndims',c)<3;
%
if isstruct(A)
	assert(isfield(A,'name'),...
		'SC:natsortfiles:A:StructMissingNameField',...
		'If first input <A> is a struct then it must have field <name>.')
	nmx = {A.name};
	assert(all(fnh(nmx)),...
		'SC:natsortfiles:A:NameFieldInvalidType',...
		'First input <A> field <name> must contain only character row vectors.')
	[fpt,fnm,fxt] = cellfun(@fileparts, nmx, 'UniformOutput',false);
	if isfield(A,'folder')
		fpt(:) = {A.folder};
		assert(all(fnh(fpt)),...
			'SC:natsortfiles:A:FolderFieldInvalidType',...
			'First input <A> field <folder> must contain only character row vectors.')
	end
elseif iscell(A)
	assert(all(fnh(A(:))),...
		'SC:natsortfiles:A:CellContentInvalidType',...
		'First input <A> cell array must contain only character row vectors.')
	[fpt,fnm,fxt] = cellfun(@fileparts, A(:), 'UniformOutput',false);
	nmx = strcat(fnm,fxt);
elseif ischar(A)
	[fpt,fnm,fxt] = cellfun(@fileparts, cellstr(A), 'UniformOutput',false);
	nmx = strcat(fnm,fxt);
else
	assert(isa(A,'string'),...
		'SC:natsortfiles:A:InvalidType',...
		'First input <A> must be a structure, a cell array, or a string array.');
	[fpt,fnm,fxt] = cellfun(@fileparts, cellstr(A(:)), 'UniformOutput',false);
	nmx = strcat(fnm,fxt);
end
%
varargin = cellfun(@ns1s2c, varargin, 'UniformOutput',false);
ixv = fnh(varargin); % char
txt = varargin(ixv); % char
xtx = varargin(~ixv); % not
%
trd = strcmpi(txt,'rmdot');
assert(nnz(trd)<2,...
	'SC:natsortfiles:rmdot:Overspecified',...
	'The "." and ".." folder handling "rmdot" is overspecified.')
%
tnx = strcmpi(txt,'noext');
assert(nnz(tnx)<2,...
	'SC:natsortfiles:noext:Overspecified',...
	'The file-extension handling "noext" is overspecified.')
%
txp = strcmpi(txt,'xpath');
assert(nnz(txp)<2,...
	'SC:natsortfiles:xpath:Overspecified',...
	'The file-path handling "xpath" is overspecified.')
%
chk = '(no|rm|x)(dot|ext|path)';
%
if nargin>1
	nsfChkRgx(rgx,chk)
	txt = [{rgx},txt(~(trd|tnx|txp))];
end
%
%% Path and Extension %%
%
% Path separator regular expression:
if ispc()
	psr = '[^/\\]+';
else % Mac & Linux
	psr = '[^/]+';
end
%
if any(trd) % Remove "." and ".." folder names
	ddx = strcmp(nmx,'.') | strcmp(nmx,'..');
	fxt(ddx) = [];
	fnm(ddx) = [];
	fpt(ddx) = [];
	nmx(ddx) = [];
end
%
if any(tnx) % No file-extension
	fnm = nmx;
	fxt = [];
end
%
if any(txp) % No file-path
	mat = reshape(fnm,1,[]);
else
	% Split path into {dir,subdir,subsubdir,...}:
	spl = regexp(fpt(:),psr,'match');
	nmn = 1+cellfun('length',spl(:));
	mxn = max(nmn);
	vec = 1:mxn;
	mat = cell(mxn,numel(nmn));
	mat(:) = {''};
	%mat(mxn,:) = fnm(:); % old behavior
	mat(logical(bsxfun(@eq,vec,nmn).')) =  fnm(:);  % TRANSPOSE bug loses type (R2013b)
	mat(logical(bsxfun(@lt,vec,nmn).')) = [spl{:}]; % TRANSPOSE bug loses type (R2013b)
end
%
if numel(fxt) % File-extension
	mat(end+1,:) = fxt(:);
end
%
%% Sort File Extensions, Names, and Paths %%
%
nmr = size(mat,1)*all(size(mat));
dbg = cell(1,nmr);
ndx = 1:numel(fnm);
%
for k = nmr:-1:1
	if nargout<3 % faster:
		[~,idx] = natsort(mat(k,ndx),txt{:},xtx{:});
	else % for debugging:
		[~,idx,gbd] = natsort(mat(k,ndx),txt{:},xtx{:});
		[~,idb] = sort(ndx);
		dbg{k} = gbd(idb,:);
	end
	ndx = ndx(idx);
end
%
% Return the sorted input array and corresponding indices:
%
if any(trd)
	tmp = find(~ddx);
	ndx = tmp(ndx);
end
%
ndx = ndx(:);
%
if ischar(A)
	B = A(ndx,:);
elseif any(trd)
	xsz = size(A);
	nsd = xsz~=1;
	if nnz(nsd)==1 % vector
		xsz(nsd) = numel(ndx);
		ndx = reshape(ndx,xsz);
	end
	B = A(ndx);
else
	ndx = reshape(ndx,size(A));
	B = A(ndx);
end
%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%natsortfiles
function nsfChkRgx(rgx,chk)
%==========================================================================
% NSFCHKRGX
%
% DESCRIPTION:
%   Validates the file-aware natural-sort regular expression.
%==========================================================================

chk = sprintf('^(%s)$',chk);
assert(~ischar(rgx)||isempty(regexpi(rgx,chk,'once')),...
	'SC:natsortfiles:rgx:OptionMixUp',...
	['Second input <rgx> must be a regular expression that matches numbers.',...
	'\nThe provided expression "%s" looks like an optional argument (inputs 3+).'],rgx)
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%nsfChkRgx
function arr = ns1s2c(arr)
%==========================================================================
% NS1S2C
%
% DESCRIPTION:
%   Converts supported string-like arrays to cell arrays of character vectors for natural sorting.
%
% OUTPUT:
%   arr : cellstr-compatible text array.
%==========================================================================

if isa(arr,'string') && isscalar(arr)
	arr = arr{1};
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ns1s2c
