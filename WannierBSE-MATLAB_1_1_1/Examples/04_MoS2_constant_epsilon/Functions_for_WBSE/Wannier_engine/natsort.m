function [B,ndx,dbg] = natsort(A,rgx,varargin)
%==========================================================================
% NATSORT
%
% DESCRIPTION:
%   Sorts text using natural-order numeric token comparison.
%
% INPUT ARGUMENTS:
%   A : values to sort.
%   rgx : optional tokenization regular expression.
%   varargin : optional sorting options.
%
% OUTPUT:
%   B : naturally sorted values.
%   ndx : sort indices.
%   dbg : tokenization diagnostics.
%==========================================================================

fnh = @(c)cellfun('isclass',c,'char') & cellfun('size',c,1)<2 & cellfun('ndims',c)<3;
%
if iscell(A)
	assert(all(fnh(A(:))),...
		'SC:natsort:A:CellInvalidContent',...
		'First input <A> cell array must contain only character row vectors.')
	B = A(:);
elseif ischar(A) % Convert char matrix:
	B = cellstr(A);
else % Convert string, categorical, datetime, etc.:
	B = cellstr(A(:));
end
%
chk = '(match|ignore)case|(de|a)scend(ing)?|(char|nan|num)[<>](char|nan|num)|%[a-z]+';
%
if nargin<2 || isnumeric(rgx)&&isequal(rgx,[])
	rgx = '\d+';
elseif ischar(rgx)
	assert(ndims(rgx)<3 && size(rgx,1)==1,...
		'SC:natsort:rgx:NotCharVector',...
		'Second input <rgx> character row vector must have size 1xN.') %#ok<ISMAT>
	nsChkRgx(rgx,chk)
else
	rgx = ns1s2c(rgx);
	assert(ischar(rgx),...
		'SC:natsort:rgx:InvalidType',...
		'Second input <rgx> must be a character row vector or a string scalar.')
	nsChkRgx(rgx,chk)
end
%
varargin = cellfun(@ns1s2c, varargin, 'UniformOutput',false);
ixv = fnh(varargin); % char
txt = varargin(ixv); % char
xtx = varargin(~ixv); % not
%
% Character case:
tcm = strcmpi(txt,'matchcase');
tcx = strcmpi(txt,'ignorecase')|tcm;
% Sort direction:
tdd = strcmpi(txt,'descend');
tdx = strcmpi(txt,'ascend')|tdd;
% Char/num order:
ttn = strcmpi(txt,'char<num');
ttx = strcmpi(txt,'num<char')|ttn;
% NaN/num order:
ton = strcmpi(txt,'NaN<num');
tox = strcmpi(txt,'num<NaN')|ton;
% SSCANF format:
tsf = ~cellfun('isempty',regexp(txt,'^%([bdiuoxfeg]|l[diuox])$'));
%
nsAssert(txt, ~(tcx|tdx|ttx|tox|tsf))
nsAssert(txt, tcx,  'CaseMatching', 'case sensitivity')
nsAssert(txt, tdx, 'SortDirection', 'sort direction')
nsAssert(txt, ttx,  'CharNumOrder', 'number<->character')
nsAssert(txt, tox,   'NanNumOrder', 'number<->NaN')
nsAssert(txt, tsf,  'sscanfFormat', 'SSCANF format')
%
% SSCANF format:
if any(tsf)
	fmt = txt{tsf};
else
	fmt = '%f';
end
%
xfh = cellfun('isclass',xtx,'function_handle');
assert(nnz(xfh)<2,...
	'SC:natsort:option:FunctionHandleOverspecified',...
	'The function handle option may only be specified once.')
assert(all(xfh),...
	'SC:natsort:option:InvalidType',...
	'Optional arguments must be character row vectors, string scalars, or a function handle.')
if any(xfh)
	txfh = xtx{xfh};
end
%
%% Identify and Convert Numbers %%
%
[nbr,spl] = regexpi(B(:), rgx, 'match','split', txt{tcx});
%
if numel(nbr)
	tmp = [nbr{:}];
	if strcmp(fmt,'%b')
		tmp = regexprep(tmp,'^0[Bb]','');
		vec = cellfun(@(s)pow2(numel(s)-1:-1:0)*sscanf(s,'%1d'),tmp);
	else
		vec = sscanf(sprintf(' %s','0',tmp{:}),fmt);
		vec = vec(2:end); % SSCANF wrong data class bug (R2009b and R2010b)
	end
	assert(numel(vec)==numel(tmp),...
		'SC:natsort:sscanf:TooManyValues',...
		'The "%s" format must return one value for each input number.',fmt)
else
	vec = [];
end
%
%% Allocate Data %%
%
% Determine lengths:
nmx = numel(B);
lnn = cellfun('length',nbr);
lns = cellfun('length',spl);
mxs = max(lns);
%
% Allocate data:
idn = logical(bsxfun(@le,1:mxs,lnn).'); % TRANSPOSE lost class bug (R2013b)
ids = logical(bsxfun(@le,1:mxs,lns).'); % TRANSPOSE lost class bug (R2013b)
arn = zeros(mxs,nmx,class(vec));
ars =  cell(mxs,nmx);
ars(:) = {''};
ars(ids) = [spl{:}];
arn(idn) = vec;
%
%% Debugging Array %%
%
if nargout>2
	mxw = 0;
	%for k = 1:nmx
	%	mxw = max(mxw,numel(nbr{k})+nnz(~cellfun('isempty',spl{k})));
	%end
	dbg = cell(nmx,mxw);
	for k = 1:nmx
		tmp = spl{k};
		if any(idn(:,k)) % Empty array allocation bug (R2009b)
			tmp(2,1:end-1) = num2cell(arn(idn(:,k),k));
		end
		tmp(cellfun('isempty',tmp)) = [];
		dbg(k,1:numel(tmp)) = tmp;
	end
end
%
%% Sort Columns %%
%
if ~any(tcm) % ignorecase
	ars = lower(ars);
end
%
if any(ttn) % char<num
	% Determine max character code:
	mxc = 'X';
	tmp = warning('off','all');
	mxc(1) = Inf;
	warning(tmp)
	mxc(mxc==0) = 255; % Octave
	% Append max character code to the split text:
	%ars(idn) = strcat(ars(idn),mxc); % slower than loop
	for k = reshape(find(idn),1,[])
		ars{k}(1,end+1) = mxc;
	end
end
%
idn(isnan(arn)) = ~any(ton); % NaN<num
%
if any(xfh) % External text-sorting function
	[~,ndx] = txfh(ars(mxs,:),txt{tcx|tdx});
	for k = mxs-1:-1:1
		[~,idx] = sort(arn(k,ndx),txt{tdx});
		ndx = ndx(idx);
		[~,idx] = sort(idn(k,ndx),txt{tdx});
		ndx = ndx(idx);
		[~,idx] = txfh(ars(k,ndx),txt{tdx|tcx});
		ndx = ndx(idx);
	end
elseif any(tdd)
	[~,ndx] = sort(nsGroups(ars(mxs,:)),'descend');
	for k = mxs-1:-1:1
		[~,idx] = sort(arn(k,ndx),'descend');
		ndx = ndx(idx);
		[~,idx] = sort(idn(k,ndx),'descend');
		ndx = ndx(idx);
		[~,idx] = sort(nsGroups(ars(k,ndx)),'descend');
		ndx = ndx(idx);
	end
else
	[~,ndx] = sort(ars(mxs,:)); % ascend
	for k = mxs-1:-1:1
		[~,idx] = sort(arn(k,ndx),'ascend');
		ndx = ndx(idx);
		[~,idx] = sort(idn(k,ndx),'ascend');
		ndx = ndx(idx);
		[~,idx] = sort(ars(k,ndx)); % ascend
		ndx = ndx(idx);
	end
end
%
%% Outputs %%
%
if ischar(A)
	ndx = ndx(:);
	B = A(ndx,:);
else
	ndx = reshape(ndx,size(A));
	B = A(ndx);
end
%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%natsort
function grp = nsGroups(vec)
%==========================================================================
% NSGROUPS
%
% DESCRIPTION:
%   Identifies numeric-token groups used by the natural-sort implementation.
%
% OUTPUT:
%   grp : numeric-token group identifiers.
%==========================================================================

[vec,idx] = sort(vec);
grp = cumsum([true(1,numel(vec)>0),~strcmp(vec(1:end-1),vec(2:end))]);
grp(idx) = grp;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%nsGroups
function nsAssert(inp,inx,ids,opt)
%==========================================================================
% NSASSERT
%
% DESCRIPTION:
%   Validates natural-sort input options and reports invalid arguments.
%==========================================================================

tmp = sprintf('The provided options:%s',sprintf(' "%s"',inp{inx}));
if nargin>2
	assert(nnz(inx)<2,...
		sprintf('SC:natsort:option:%sOverspecified',ids),...
		'The %s option may only be specified once.\n%s',opt,tmp)
else
	assert(~any(inx),...
		'SC:natsort:option:InvalidOptions',...
		'Invalid options provided. Check the help and option spelling!\n%s',tmp)
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%nsAssert
function nsChkRgx(rgx,chk)
%==========================================================================
% NSCHKRGX
%
% DESCRIPTION:
%   Validates the regular expression used for natural-sort numeric tokenization.
%==========================================================================

chk = sprintf('^(%s)$',chk);
assert(isempty(regexpi(rgx,chk,'once')),...
	'SC:natsort:rgx:OptionMixUp',...
	['Second input <rgx> must be a regular expression that matches numbers.',...
	'\nThe provided input "%s" looks like an optional argument (inputs 3+).'],rgx)
if isempty(regexpi('0',rgx,'once'))
	warning('SC:natsort:rgx:SanityCheck',...
		['Second input <rgx> must be a regular expression that matches numbers.',...
		'\nThe provided regular expression does not match the digit "0", which\n',...
		'may be acceptable (e.g. if literals, quantifiers, or lookarounds are used).'...
		'\nThe provided regular expression: "%s"'],rgx)
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%nsChkRgx
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
