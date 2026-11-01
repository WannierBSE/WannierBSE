function reset_internal_kmesh_tb_cache(kmeshCacheFile, tbCachePath, TempTopologyPath, removeKmesh)
%==========================================================================
% RESET_INTERNAL_KMESH_TB_CACHE
%
% DESCRIPTION:
%   Deletes stale internally generated k-mesh/TB cache files so WBSE can rebuild a coherent cache set.
%
% INPUT ARGUMENTS:
%   kmeshCacheFile : generated kmesh file path.
%   tbCachePath : generated TB cache directory.
%   TempTopologyPath : generated topology sidecar directory.
%   removeKmesh : true to also remove the kmesh file.
%
% OUTPUT:
%   None. Selected cache files and directories are removed and recreated.
%==========================================================================

    if removeKmesh && isfile(kmeshCacheFile)
        delete(kmeshCacheFile);
    end
    if isfolder(tbCachePath)
        rmdir(tbCachePath, 's');
    end
    mkdir(tbCachePath);
    if removeKmesh && isfolder(TempTopologyPath)
        rmdir(TempTopologyPath, 's');
    end
end
