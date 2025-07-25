import os

import numpy as np
import pandas as pd
from scipy import spatial
import trimesh
import alphashape

import navis.interfaces.neuprint as neu
from neuprint import NeuronCriteria as NC, SynapseCriteria as SC, fetch_synapses
from utils.helper import get_data_path

def create_edge_ids(
    roi_str = 'ME(R)'
) -> None:
    """
    Find edge coordinates of pins and store in, e.g. ME_hex_ids_edge.csv

    Parameters
    ----------
    syn_df : pd.DataFrame
        dataframe with 'bodyId', 'x', 'y', 'z' columns
    roi_str : str
        neuprint ROI, can only be ME(R), LO(R), LOP(R)
    samp : int
        sub-sampling factor for depth bins

    Returns
    -------
    syn_col_df : pd.DataFrame
        'hex1_id' : int
            defines column
        'hex2_id' : int
            defines column
        'col_count' : int
            number of synapses in that column (across all depth bins)
    """

    data_path = get_data_path(reason='cache')

    pincushion_f = data_path / f"{roi_str[:-3]}_col_center_pins.pickle"
    with open(pincushion_f, mode='rb') as pc_fh:
        pin_cushion_df = pd.read_pickle(pc_fh)
    pin_cushion_df = pin_cushion_df.dropna()

    hex_edge_df = get_edge_ids(pin_cushion_df)

    hex_edge_f = data_path / f"{roi_str[:-3]}_hex_ids_edge.csv"
    hex_edge_df.to_csv(hex_edge_f, index=True, header=False)


def get_edge_ids(
    df:pd.DataFrame
) -> pd.DataFrame:
    """
    Find the hex ids of the edges

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that has `['hex1_id', 'hex2_id']` columns.

    Returns
    -------
    hex_edge_df : pd.DataFrame
        DataFrame with `['hex1_id', 'hex2_id']` of the edges
        index is the subset of indeces of df that are on the edge
    """
    df = df\
        .drop_duplicates(['hex1_id', 'hex2_id'])[['hex1_id', 'hex2_id']]\
        .astype(int)
    df['neighbors'] = df.apply(
        lambda r: np.where(
            (np.abs(df['hex1_id']-r.hex1_id)==1)
           &(np.abs(df['hex2_id']-r.hex2_id)==1))[0]
      , axis=1)
    df['is_edge'] = df.apply(lambda r: len(r['neighbors'])<4, axis=1)
    hex_edge_df = df[df['is_edge']][['hex1_id', 'hex2_id']]

    return hex_edge_df


def load_pins_for_mesh(roi_str):
    """
    Load data for layer mesh creation

    Parameters
    ----------
    roi_str : str
        name of the neuropil, can only be ME(R), LO(R), LOP(R)

    Returns
    -------
    pins : np.ndarray
        xyz positions of pin nodes (excl pins on the edge without 2 neighbors)
    n_bins : int
        number of depth bins (same for all pins)
    hex1_valid : np.ndarray
        hex1_ids of included pins
    hex2_valid : np.ndarray
        hex2_ids of included pins
    """

    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)'],\
            f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)', but is actually '{roi_str}'"

    data_path = get_data_path(reason='cache')
    col_ids, _, n_bins, pins = load_pins(roi_str=roi_str)
    pins = pins.astype(float)

    if (roi_str=='LO(R)')|(roi_str=='LOP(R)'):
        #manual exclusion of boundary pins that don't have 2 neighbors
        if roi_str=='LOP(R)':
            col_exclude = [0,2,36,37,156,853,883,888,889]
        elif roi_str=='LO(R)':
            col_exclude = [0]
        id_exclude = np.where( np.isin( col_ids, col_exclude) )[0]
        id_include = np.array([i for i in range(col_ids.shape[0]) if i not in id_exclude])
        col_ids = col_ids[id_include]
        pins_sorted = pins.reshape((-1,n_bins,3))
        pins_sorted = pins_sorted[id_include]
        pins = pins_sorted.reshape((-1,3))
    hex_df = pd.read_pickle(data_path / 'ME_col_center_pins.pickle')[['hex1_id','hex2_id']]\
        .astype('int')
    hex1_valid = hex_df['hex1_id'].values
    hex2_valid = hex_df['hex2_id'].values
    #the (1,10) column is effectively a (1,9) column (which doesn't exist)
    hex2_valid[2] = 9
    hex1_valid = hex1_valid[col_ids]
    hex2_valid = hex2_valid[col_ids]

    #find which pins are on the edge
    col_ids_edge_f = data_path / f"{roi_str[:-3]}_hex_ids_edge.csv"
    if not col_ids_edge_f.is_file():
        create_edge_ids(roi_str=roi_str)
    col_ids_edge_df = pd.read_csv(col_ids_edge_f, header=None)
    col_ids_edge = col_ids_edge_df.values[:,0]
    if (roi_str=='LO(R)')|(roi_str=='LOP(R)'):
        id_exclude = np.where( np.isin( col_ids_edge, col_exclude) )[0]
        id_include = np.array([i for i in range(col_ids_edge.shape[0]) if i not in id_exclude])
        col_ids_edge = col_ids_edge[id_include]
    idx_edge = np.where(np.isin(col_ids, col_ids_edge))[0]

    return pins, n_bins, hex1_valid, hex2_valid, idx_edge


def make_large_mesh(roi_str, bin_delta=1
) -> None :
    """
    Create obj files with large (larger than neurpil rois) layer rois

    The files (named roi_str[:-3]+'_layer_'+str(l+1)+'_L') save a trimesh

    Parameters
    ----------
    roi_str : str
        name of the neuropil, can only be ME(R), LO(R), LOP(R)
    bin_delta : int
        determines thickness of surface on the edge
        edge points are chosen with depth within +-bin_delta bins of the layer threshold

    Returns
    -------
    None
    """

    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)'],\
            f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)', but is actually '{roi_str}'"

    alpha, fac_ext = load_roi_layer_mesh_params(roi_str=roi_str)

    data_path = get_data_path(reason='cache')
    pins, n_bins, hex1_valid, _, idx_edge = load_pins_for_mesh(roi_str=roi_str)
    n_col = hex1_valid.shape[0]

    #find threshold on bins to define layers
    depth_bdry = load_layer_thre(roi_str=roi_str)
    bin_bdry = n_bins-1-np.round(depth_bdry*(n_bins-1)).astype(int)
    bin_bdry[-1] = 0
    bin_bdry[0] = n_bins-1
    n_layers = bin_bdry.shape[0]-1

    #get surface points of neuropil and find their depth
    neu_vol = neu.fetch_roi(roi_str)
    roi_xyz = neu_vol.vertices
    depth_df = find_depth(pd.DataFrame(roi_xyz, columns=['x','y','z']), roi_str=roi_str, samp=1)

    #find points on top surface
    tree = spatial.KDTree(roi_xyz)
    pins0 = pins[bin_bdry[0]::n_bins]
    neu_dists0 = trimesh.proximity.signed_distance(neu_vol,pins0)
    ind_replace0 = np.where( neu_dists0<=0 )[0]
    _, minid0 = tree.query(pins0)
    for i in range(n_col):
        if (i in ind_replace0):
            pins0[i] = roi_xyz[minid0[i]]

    #find points on the boundary in top layer (making sure not at the very top)
    pins0_edge = pins0[idx_edge].copy()
    roi_xyz_bdry = roi_xyz[(depth_df['bin']>=bin_delta)&(depth_df['bin']<3*bin_delta)]
    tree_edge = spatial.KDTree(roi_xyz_bdry)
    _, minid0 = tree_edge.query(pins0_edge)
    pins0_edge = roi_xyz_bdry[minid0]
    pins0_edge = pins0_edge + fac_ext*(pins0_edge-pins0[idx_edge])

    #find points on the boundary in each layer
    for l in range(n_layers-1):
        pins1 = pins[bin_bdry[l+1]::n_bins]
        pins1_edge = pins1[idx_edge].copy()
        roi_xyz_bdry = roi_xyz[
            (depth_df['bin'] >= n_bins - 1 - bin_bdry[l+1] - bin_delta)\
          & (depth_df['bin'] < n_bins - 1 - bin_bdry[l+1] + bin_delta)
        ]
        tree_edge = spatial.KDTree(roi_xyz_bdry)
        _, minid1 = tree_edge.query(pins1_edge)
        pins1_edge = roi_xyz_bdry[minid1]
        pins1_edge = pins1_edge + fac_ext*(pins1_edge-pins1[idx_edge])

        #use points to define layer
        layer_pts = np.concatenate((pins0, pins0_edge, pins1, pins1_edge), axis=0)
        shell = alphashape.alphashape(layer_pts, alpha=alpha)
        trimesh.repair.fix_inversion(shell)
        trimesh.repair.fix_normals(shell)
        trimesh.repair.fix_winding(shell)
        shell.export(data_path / f"{roi_str[:-3]}_layer_{l+1}_L.obj")

        #make lower surface of current layer to be the upper surface of next layer
        pins0 = pins1
        pins0_edge = pins1_edge

    #find points on bottom surface
    pins1 = pins[bin_bdry[-1]::n_bins]
    neu_dists1 = trimesh.proximity.signed_distance(neu_vol,pins1)
    ind_replace1 = np.where( neu_dists1<=0 )[0]
    _, minid1 = tree.query(pins1)
    for i in range(n_col):
        if i in ind_replace1:
            pins1[i] = roi_xyz[minid1[i]]

    #find points on the boundary in bottom layer (making sure not at the very bottom)
    pins1_edge = pins1[idx_edge].copy()
    roi_xyz_bdry = roi_xyz[
        (depth_df['bin'] >= n_bins - 1 - 3 * bin_delta) \
      & (depth_df['bin'] < n_bins - 1 - bin_delta)
    ]
    tree_edge = spatial.KDTree(roi_xyz_bdry)
    _, minid1 = tree_edge.query(pins1_edge)
    pins1_edge = roi_xyz_bdry[minid1]
    pins1_edge = pins1_edge + fac_ext*(pins1_edge-pins1[idx_edge])

    #use points to define bottom layer
    layer_pts = np.concatenate((pins0, pins0_edge, pins1, pins1_edge), axis=0)
    shell = alphashape.alphashape(layer_pts, alpha=alpha)
    trimesh.repair.fix_inversion(shell)
    trimesh.repair.fix_normals(shell)
    trimesh.repair.fix_winding(shell)
    shell.export(data_path / f"{roi_str[:-3]}_layer_{n_layers}_L.obj")


def load_roi_layer_mesh_params(roi_str='ME(R)'):
    """
    Load parameters for layer mesh creation.

    All parameters were chosen heuristically.

    Parameters
    ----------
    roi_str : str, default='ME(R)'
        neuprint ROI, can only be ME(R), LO(R), LOP(R)

    Returns
    -------
    alpha : float
        used in alphashape; smaller numbers correspond to more smoothing
    fac_ext : float
        extension parameter of edge pin points towards the boundary
    """
    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)'],\
            f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)', but is actually '{roi_str}'"

    data_path = get_data_path(reason='params')
    params_fn = data_path / "layer_mesh_creation_parameters.xlsx"
    params_df = pd.read_excel(params_fn).convert_dtypes()
    params_df = params_df[ params_df['neuropil']==roi_str[:-3] ]

    #parameter alpha is used in alphashape; smaller numbers correspond to more smoothing
    #parameter fac_ext determines how much edge column points are extended towards the boundary,
    alpha = float( params_df[ params_df['parameter']=='alpha' ]['value'].values[0] )
    fac_ext = float( params_df[ params_df['parameter']=='fac_ext' ]['value'].values[0] )

    return alpha, fac_ext


def create_ol_layer_boundaries(
    rois:list[str]=None
) -> None:
    """
    Create the boundary files for all optic lobe neuropils (currently ME/LO/LOP).

    Parameters
    ----------
    rois : list[str], default=None
        If `rois` is None, it uses ['ME(R)', 'LO(R)', 'LOP(R)'].

    Returns
    -------
    None
    """
    # Following recommendation
    # https://pylint.pycqa.org/en/latest/user_guide/messages/warning/dangerous-default-value.html
    if rois is None:
        rois = ['ME(R)', 'LO(R)', 'LOP(R)']
    for roi in rois:
        find_layer_bdrys(roi)


def find_layer_bdrys(
    roi_str:str='ME(R)'
) -> None:
    """
    Find the depth thresholds between layers (very top and very bottom are set manually).

    It generates files in the data path named `ME_layer_bdry.csv`, `LO_layer_bdry.csv`,
        or `LOP_layer_bdry.csv`.

    Parameters
    ----------
    roi_str : str, default='ME(R)'
        neuprint neuropil name, can only be ME(R), LO(R), LOP(R)

    Returns
    -------
    None
    """

    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)'],\
            f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)', but is actually '{roi_str}'"

    data_path = get_data_path(reason='cache')

    n_layers, frac_peaks, canonical_list, syn_type, peak_ids, layer_bdry_pos =\
        load_roi_layer_params(roi_str=roi_str)
    bin_edges, bin_centers = load_depth_bins(roi_str=roi_str, samp=1)

    layers_bdry = np.zeros(n_layers+1)
    layers_bdry[0] = -0.01
    layers_bdry[-1] = 1.01
    for idx, target_type in enumerate(canonical_list):
        syn_df = fetch_synapses(
            NC(type=target_type)
          , SC(rois=roi_str, confidence=.9))

        syn_df = syn_df[syn_df['type']==syn_type[idx]]
        depth_df = find_depth(syn_df, roi_str=roi_str, samp=1)

        count, _ = np.histogram(
            depth_df['depth'].values
          , bins=bin_edges
          , density=True)
        thre_peaks = find_pdf_treshold(count, frac_peaks)
        depth_bottom, depth_top = find_peak_thresholds(
            count
          , bin_centers
          , thre_peaks
          , min_bin_change=5)

        depth_bdrys = np.vstack([depth_bottom, depth_top]).T.reshape((-1,1))
        layers_bdry[layer_bdry_pos[idx]] = np.squeeze(depth_bdrys[peak_ids[idx]])

    #in the LOP, there are gaps between the T4 layer patterns
    #therefore, we define the layer boundaries as means of two neighboring estimates
    if roi_str=='LOP(R)':
        only_four_layers_bdry = np.zeros(5)
        only_four_layers_bdry[[0,-1]] = layers_bdry[[0,-1]]
        only_four_layers_bdry[1] = layers_bdry[1:3].mean()
        only_four_layers_bdry[2] = layers_bdry[3:5].mean()
        only_four_layers_bdry[3] = layers_bdry[5:7].mean()
        layers_bdry = only_four_layers_bdry

    pd.DataFrame(layers_bdry).to_csv(
        data_path / f"{roi_str[:-3]}_layer_bdry.csv"
      , float_format='%.2f'
      , index=False
      , header=False)


def load_roi_layer_params(roi_str='ME(R)'):
    """
    Load parameters for layer creation.

    All parameters were chosen heuristically.

    Parameters
    ----------
    roi_str : str, default='ME(R)'
        neuprint ROI, can only be ME(R), LO(R), LOP(R)

    Returns
    -------
    n_layers : int
        number of layers
    frac_peaks : float
        fraction of synapses in all peaks
    canonical_list : list[str]
        cell-types to define layer boundaries
    syn_type : list[str]
        synapse type to define layer boundaries
    peak_ids : list[list[int]]
        position of peaks that are used to define layer boundaries.
        length of the outer list equals the length of `canonical_list`.
        length of the inner list equals the number of flanks of peaks that are used to define
            layer boundaries. The numbers mean: `0` is the left flank of the first peak, `1` the
            right flank of the first peak, `2` the left flank of the third peak, etc…
    layer_bdry_pos : list[list[int]]
        position of flanks in layer boundary list.
        length of the outer list equals the length of `canonical_list`.
        length of the inner list equals the number of flanks of peaks that are used to define
            layer boundaries. The numbers mean: `1` is the layer threshold between layers 1 and 2,
            `2` is the layer threshold between layers 2 and 3, etc…
    """
    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)'],\
            f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)', but is actually '{roi_str}'"

    data_path = get_data_path(reason='params')
    params_fn = data_path / "layer_creation_parameters.xlsx"
    params_df = pd.read_excel(params_fn).convert_dtypes()
    params_df = params_df[ params_df['neuropil']==roi_str[:-3] ]

    n_layers = int(params_df[ params_df['parameter']=='n_layers' ]['value'].values[0])
    frac_peaks = float(params_df[ params_df['parameter']=='frac_peaks' ]['value'].values[0])
    canonical_list = params_df[ params_df['parameter']=='canonical_list' ]['value']\
        .values[0]\
        .split(", ")
    syn_type = params_df[ params_df['parameter']=='syn_type' ]['value']\
        .values[0]\
        .split(", ")
    peak_ids = (
            params_df[ params_df['parameter']=='peak_ids' ]['value']\
                .apply(convert_str_to_list_of_lists_of_ints)
        ).values[0]
    layer_bdry_pos = (
            params_df[ params_df['parameter']=='layer_bdry_pos' ]['value']\
                .apply(convert_str_to_list_of_lists_of_ints)
        ).values[0]

    return n_layers, frac_peaks, canonical_list, syn_type, peak_ids, layer_bdry_pos


def convert_str_to_list_of_lists_of_ints(
    list_of_lists:str
) -> list:
    """
    Helper function.
    Converts a string that is a list of lists of integers to that object.

    Parameters
    ----------
    list_of_lists : str
        string containing the list of lists

    Returns
    -------
    list_all : list[list[int]]
        list of lists of integers contained in s
    """
    s_split = list_of_lists.split(',')
    list_all = []
    idx = 0
    while idx < len(list_of_lists.split(',')):
        tmp_list = []
        while s_split[idx][-1]!=']':
            tmp_list.append( int(s_split[idx].strip('[]')) )
            idx += 1
        tmp_list.append( int(s_split[idx].strip('[]')) )
        list_all.append(tmp_list)
        idx += 1

    return list_all


def find_pdf_treshold(
    count
  , frac_peaks
  , thre_peaks_step=0.1
):
    """
    Find threshold on synapse density to reach fraction `frac_peaks` of total,
        e.g. `frac_peaks=0.8` means that we are looking for the threshold on the synapse density
        to reach 80% of all synapses

    Parameters
    ----------
    count : np.ndarray
        synapse distribution
    frac_peaks : float
        fraction of synapses in all peaks
    thre_peaks_step : float
        step size in count to reach fraction `frac_peaks`

    Returns
    -------
    thre_peaks : float
        threshold on count to find peaks
    """
    thre_peaks = count.max()
    count_all = count.sum()
    count_peaks = count[ count>=thre_peaks ].sum()
    while count_peaks/count_all < frac_peaks:
        thre_peaks = thre_peaks-thre_peaks_step
        count_peaks = count[ count>=thre_peaks ].sum()

    return thre_peaks


def find_peak_thresholds(
    count
  , bin_centers
  , thre_peaks
  , min_bin_change=2
):
    """
    Find the flanks of peaks in the synapse distribution

    Parameters
    ----------
    count : np.ndarray
        synapse distribution
    bin_centers : np.ndarray
        bin centers for depth
    thre_peaks : float
        threshold on synapse distribution to find peaks
    min_bin_change : int
        minimum number of bins that top and bottom of neighboring peaks have to be separated by

    Returns
    -------
    depth_bottom : np.ndarray
        sub-array of bin_centers with the bottom flanks of peaks
    depth_top : np.ndarray
        sub-array of bin_centers with the top flanks of peaks
    """

    #Find peaks
    ind_peak = np.where(count >= thre_peaks)[0]
    diff_ind = np.diff(ind_peak, 1)
    ind_const = np.where(diff_ind <= min_bin_change)[0]
    ind_change = np.where(diff_ind > min_bin_change)[0]
    #how many different peaks
    k = ind_change.shape[0] + 1
    #find inidices that define bottom and top of peak
    ind_bottom = np.zeros(k, dtype=int)
    ind_top = np.zeros(k, dtype=int)
    #find corresponding depths
    depth_bottom = np.zeros(k)
    depth_top = np.zeros(k)
    ind_bottom[0] = ind_peak[ind_const[0]]
    depth_bottom[0] = bin_centers[ind_bottom[0]]
    #loop through peaks
    for i in range(k-1):
        if (ind_peak[ind_change] > ind_bottom[i]).sum() > 0:
            ind_top[i] = ind_peak[ind_change][np.where(ind_peak[ind_change]>ind_bottom[i])[0][0]]
        else:
            ind_top[i] = bin_centers.shape[0]-1
        depth_top[i] = bin_centers[ind_top[i]]
        if (ind_peak[ind_const]>ind_top[i]).sum()>0:
            ind_bottom[i+1] = ind_peak[ind_const][np.where(ind_peak[ind_const]>ind_top[i])[0][0]]
        else:
            ind_bottom[i+1] = 0
        depth_bottom[i+1] = bin_centers[ind_bottom[i+1]]
    ind_top[k-1] = ind_peak[-1]
    depth_top[k-1] = bin_centers[ind_top[k-1]]

    return depth_bottom, depth_top


# Needs to be at the end because of circular import
from utils.ROI_calculus import\
    find_depth, load_layer_thre, load_pins, load_depth_bins
