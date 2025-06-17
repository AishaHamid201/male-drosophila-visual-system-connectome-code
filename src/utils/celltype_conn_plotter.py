from abc import ABC
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.colors as pc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.celltype_conn_by_roi import CelltypeConnByRoi

class CelltypeConnPlotter(ABC):

    """
    This class is used to plot basic exploratory plots from the CelltypeConnByRoi class
    It is designed to contain all the plots in a dictionary with the plot title being the key and
    the value being the figure itself.

    All plotting function also have a logical flag to plot directly to screen
    """


    # common values for all the instantiations of the class
    __colormap_seq = 'YlOrRd_r'
    __color_bar = '#3283FE'
    __color_bar2 = '#F6222E'
    __color_background = 'rgb(203,213,232)'


    def __init__(self, ctype_cbyr:CelltypeConnByRoi):
        assert isinstance(ctype_cbyr, CelltypeConnByRoi),\
            'input must be an CelltypeConnByRoi object'
        self.__ctype_connbyroi = ctype_cbyr
        self.__fig_dict: dict = {}


    def get_ctype_connbyroi(self):
        return self.__ctype_connbyroi


    def get_fig_dict(self):
        return self.__fig_dict


    def plot_from_fig_dict(self):
        temp_fig_dict = self.get_fig_dict()
        fig_tit = temp_fig_dict.keys()
        if len(fig_tit) == 0:
            raise ValueError("No figures to in dictionary")

        tit_dict={}
        for ind, key in enumerate(fig_tit):
            tit_dict[ind] = key

        print("Please enter a number for the desired plot: \n")
        print("\n".join(f"{k}\t{v}" for k, v in tit_dict.items()))
        val = input('Choose figure to plot:')

        rel_key = tit_dict[int(val)]
        rel_fig = temp_fig_dict[rel_key]

        rel_fig.show()


    def plot_all_figs(self):
        fig_dict = self.get_fig_dict()

        for fig_tit, fig_plot in fig_dict.items():
            print(fig_tit)
            fig_plot.show()


    def plot_consistent_connections_by_type(
        self
      , conn_dir: str
      , roi_to_plot: str=None
      , show_flag: bool=True
    ):
        """"
        Plots 2 scatter plots summarizing all the connected pre_type connections

        1. fraction of post_cells that have a connection with a specific pre_type vs. median
          fraction of input from that pre-type
        2. sorted fraction of post_cells that have a connection with a specific pre_type (same
          variable as above)

        Parameters
        ----------
        conn_dir : str
            'input' : when plotting all the different input celltypes
            'output' : when plotting all the different output celltypes
        roi_to_plot : str, default=None
            giving the name of a particular ROI will plot data only from that ROI, if not given plots all data
        show_flag : bool, default=True
            if true adds the figure to the fig_dict and also plots to screen

        """
        ctype_cbyr = self.get_ctype_connbyroi()

        assert conn_dir in ['input', 'output'], 'conn_dir should be either input or output'

        if conn_dir == 'input':

            if ctype_cbyr.get_input_celltypes_w_stats() is None:
                ctype_cbyr.calc_input_celltypes_w_stats()

            orig_ct_df = ctype_cbyr.get_input_celltypes_w_stats()
            dir_type = 'pre'
            tit1 = 'frac post cells with <br>pre_type'
            tit2 = 'frac of input from <br>pre_type'
        else:
            if ctype_cbyr.get_output_celltypes_w_stats() is None:
                ctype_cbyr.calc_output_celltypes_w_stats()

            orig_ct_df = ctype_cbyr.get_output_celltypes_w_stats()
            dir_type = 'post'
            tit1 = 'frac pre cells with <br>post_type'
            tit2 = 'frac of output to <br>post_type'

        av_rois = orig_ct_df['roi'].unique()
        if roi_to_plot:
            assert roi_to_plot in av_rois, f"ROI should be from the following: {av_rois} or None for ALL"

        conn_type = f'type_{dir_type}'
        frac_type = f'frac_tot_{dir_type}_roi_tot'
        tot_syn_type = f'tot_syn_per_{dir_type}Nroi'

        if roi_to_plot:
            conn_ct_df = orig_ct_df[orig_ct_df['roi'].eq(roi_to_plot)].copy()
            conn_ct_df['norm_type_frac'] = conn_ct_df['type_frac']
            conn_ct_df['norm_type_counter'] = conn_ct_df['type_counter']
        else:
            temp_ct_df = orig_ct_df.groupby(conn_type, as_index=False)[[frac_type, tot_syn_type, 'type_frac', 'type_counter']].sum()
            temp_roi_df = orig_ct_df.groupby(conn_type, as_index=False)['roi'].count()
            conn_ct_df = temp_ct_df.merge(temp_roi_df, on = conn_type)
            conn_ct_df['norm_type_frac'] = conn_ct_df['type_frac'].div(conn_ct_df['roi'])
            conn_ct_df['norm_type_counter'] = conn_ct_df['type_counter'].div(conn_ct_df['roi'])

        conn_ct_df = conn_ct_df.sort_values(by='norm_type_frac', ascending=False)

        # generating the right strings for hovering

        type_str = conn_ct_df[conn_type].values
        nums = [int(num) for num in conn_ct_df['norm_type_counter'].values]
        fracs = list(conn_ct_df['norm_type_frac'].values)

        num_str = [f"{number:.0f}" for number in nums]
        syn_str = [f"{syn_num:.0f}" for syn_num in conn_ct_df[tot_syn_type].values]
        frac_str = [f"{number:.2f}" for number in fracs]

        point_txt1 = []
        for cell_type, num, syn_num in zip(type_str, num_str, syn_str):
            point_txt1.append(f"{cell_type}<br>Count:{num}<br>Med_syn_num:{syn_num}")

        point_txt2 = []
        for cell_type, num, syn_num in zip(type_str, frac_str, syn_str):
            point_txt2.append(f"{cell_type}<br>Frac:{num}<br>Med_syn_num:{syn_num}")

        # plotting the figure

        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.15)

        fig.add_trace(
            go.Scatter(
                x=conn_ct_df['norm_type_frac'], y=conn_ct_df[frac_type],
                hoverinfo='text', hovertext=point_txt1, mode='markers',
                marker={
                    'color': self.__color_bar,
                    'size': 6},
                showlegend=False),
            row=1, col=1)

        fig.add_trace(
            go.Scatter(
                x=conn_ct_df[conn_type], y=conn_ct_df['norm_type_frac'],
                hoverinfo='text', hovertext=point_txt2, mode='markers',
                marker={
                    'color':self.__color_bar,
                    'size': 4 * pow(conn_ct_df[tot_syn_type].values, 0.5)},
                showlegend=False),
            row=2, col=1)

        fig.update_xaxes(showgrid=False)
        fig.update_xaxes(title_text=tit1, row=1, col=1)
        fig.update_yaxes(title_text=tit2, row=1, col=1)
        fig.update_yaxes(title_text=tit1, row=2, col=1)

        fig.update_layout(
            font={'size': 12},
            width=1000, height=700
        )

        self.__fig_dict[f'plot_consistent_connections_by_type:{conn_dir}'] = fig

        if show_flag:
            fig.show()


    def plot_sorted_boxplots_by_conn_type(
        self
      , conn_dir: str
      , prop_to_plot: str
      , roi_to_plot: str=None
      , box_orient_vert: bool=True
      , log_flag: bool=False
      , frac_thresh: float=0.5
      , med_thresh: float=0.01
      , show_flag: bool=True
    ) -> np.ndarray:
        """
        Plots the inputs or outputs to a cell type sorted on the median fraction of that type from
          total inputs to each cell. Note cell types that only contact 1/2 of the population of 
          target_type are excluded from the analysis

        Parameters
        ----------
        conn_dir : str
            'input' : when plotting all the different input celltypes.
            'output' : when plotting all the different output celltypes.
        prop_to_plot : str
            whether the plot is of fraction of connections the connected types provides or simply
              the number of synapses
            string should be either 'frac' or 'syn'
        roi_to_plot : str=None
            giving the name of a particular ROI will plot data only from that ROI, if not given
            plots all data when roi is given prop_to_plot is plotted it from its 'per_roi' version
        box_orient_vert : bool, default=True
            whether to plot the figure horizontally or vertically. meant to deal with large 
              numbers of pre_types
        log_flag : bool=False
            If True plots the value axes on a log scale
        frac_thresh : float, default=0.5
            Only cell types who connect to at least threshold fraction the target type will be
              included in the plot
        med_thresh : float, default=0.01
            Only cells that provide med_thresh fraction of the inputs to the target type will be 
              plotted
        show_flag : bool (default=True)
            if true adds the figure to the fig_dict and also plots to screen

        Returns
        -------
        box_order.values : numpy.ndarray
            Array of cell-types ordered as plotted (High to low)
        """

        ctype_cbyr = self.get_ctype_connbyroi()

        assert conn_dir in ['input', 'output'], 'conn_dir should be either input or output'
        assert prop_to_plot in ['frac', 'syn'], 'prop_to_plot should be either frac or syn'

        if conn_dir == 'input':
            conn_ct_df = ctype_cbyr.get_input_neurons_w_stats()
            dir_type = 'pre'
            rel_bod = 'bodyId_post'
        else:
            conn_ct_df = ctype_cbyr.get_output_neurons_w_stats()
            dir_type = 'post'
            rel_bod = 'bodyId_pre'

        conn_type = f'type_{dir_type}'
        frac_type = f'frac_tot_{dir_type}'
        syn_type = f'tot_syn_per_{dir_type}'
        prop_str_add = ''
        prop_frac_add = ''

        av_rois = conn_ct_df['roi'].unique()
        if roi_to_plot:
            assert roi_to_plot in av_rois, f"ROI should be from the following: {av_rois} or None for ALL"
            conn_ct_df = conn_ct_df[conn_ct_df['roi'].eq(roi_to_plot)]
            prop_str_add = 'Nroi'
            prop_frac_add = '_by_roi'

        if prop_to_plot == 'frac':
            prop_name = frac_type + prop_frac_add
        else:
            prop_name = syn_type + prop_str_add

        crit = conn_ct_df['rank_dense'] == 1
        temp_df = conn_ct_df[crit]
        # before calculating the median, removing cell_types that appear fewer than half the time
        num_cells_df = temp_df\
            .groupby([conn_type])[rel_bod]\
            .unique()\
            .transform(lambda x: x.shape[0])
        crit2 = num_cells_df > num_cells_df.max() * frac_thresh
        type_sel_df = temp_df\
            .merge(crit2, 'right', on=conn_type, suffixes=('', '_bool'))
        type_sel_df\
            .rename(
                columns={f'{rel_bod}_bool': f'{conn_type}_bool'},
                inplace=True) # no real reason to do this
        type_sel_df = type_sel_df[type_sel_df[f'{conn_type}_bool']]

        med_df = type_sel_df\
            .groupby([conn_type])[frac_type]\
            .transform('median')
        temp_df2 = type_sel_df[med_df.values > med_thresh]

        box_order = temp_df2\
            .groupby([conn_type])[prop_name]\
            .median()\
            .sort_values(ascending=False)\
            .index

        fig = go.Figure()

        if box_orient_vert:
            for ctype in box_order:
                rel_v = temp_df2[temp_df2[conn_type] == ctype][prop_name]
                fig.add_trace(go.Box(y=rel_v, name=ctype, marker_color=self.__color_bar))
        else:
            for ctype in box_order:
                rel_v = temp_df2[temp_df2[conn_type] == ctype][prop_name]
                fig.add_trace(go.Box(x=rel_v, name=ctype, marker_color=self.__color_bar))
                fig.layout.yaxis.autorange = "reversed"

        fig.layout.update(showlegend=False)

        if log_flag:
            if box_orient_vert:
                fig.update_yaxes(type='log')
            else:
                fig.update_xaxes(type='log')

        self.__fig_dict[f"plot_sorted_boxplots_by_conn_type:{conn_dir} {prop_name}"] = fig

        if show_flag:
            fig.show()

        return box_order.to_list()


    def plot_neuron_properties_by_spatial_position(
        self
      , conn_dir : str
      , prop_to_plot : str
      , roi_to_plot : str=None
      , min_num_syn : int=1
      , frac_thresh : float=0.75
      , prop_thresh : int=None
      , show_flag : bool=True
    ):
        """
        This functon plots a neuronal properties by the position of the neurons average position of
            its input(output) synapses

        Parameters
        ----------
        conn_dir : str
            'input' : when plotting all the different input celltypes.
            'output' : when plotting all the different output celltypes.
        prop_to_plot : str
            property to plot, can be one of 3 (for now)
            'syn' : total number of synapses
            'conn' : total number of connected cells
            'num_to_thresh' : total number of cell types to reach a fraction of connected threshold
        roi_to_plot : str, default=None
            giving the name of a particular ROI will plot data only from that ROI, if there are multiple ROIs this input is neccessary
            if there is only one ROI in the object that will be plotted
        min_num_syn : int, default=1
            adjusts both 'tot_syn' and 'tot_conn' plots. minimum number of synapses to consider a
              connection
        frac_thresh : float, default=0.75
            adjusts only the 'num_to_thresh' plot. threshold for fraction of input or output to
              reach
        prop_thresh : int, default=None
            if given turns the plot into a binary plot, with values above and below the given
              number plotted in 2 different colors (applies to all plotted properties)
        show_flag : bool, default=True
            if true adds the figure to the fig_dict and also plots to screen
        
        Returns:
        --------
        fig: matplotlib.pyplot.figure
        """
        ctype_cbyr = self.get_ctype_connbyroi()

        assert conn_dir in ['input', 'output'], 'conn_dir should be either input or output'
        assert prop_to_plot in ['syn', 'conn', 'num_to_thresh'],\
            'prop_to_plot should be syn, conn or num_to_thresh'

        if conn_dir == 'input':

            conn_ct_df = ctype_cbyr.get_input_neurons_w_stats()

            rel_frac = 'frac_tot_pre'
            rel_bid = 'bodyId_post'
            rel_tot = 'tot_num_inp'
        else:

            conn_ct_df = ctype_cbyr.get_output_neurons_w_stats()

            rel_frac = 'frac_tot_post'
            rel_bid = 'bodyId_pre'
            rel_tot = 'tot_num_out'

        av_rois = conn_ct_df['roi'].unique()

        if (not isinstance(av_rois, str)) & (len(av_rois) > 1):  
            if roi_to_plot is None:
                raise ValueError(
                    f"an ROI must be selected from {av_rois}")
            else:
                assert roi_to_plot in av_rois, f"ROI should be from the following: {av_rois}"
                conn_ct_df = conn_ct_df[conn_ct_df['roi'].eq(roi_to_plot)]
        elif isinstance([av_rois], str):  
            conn_ct_df = conn_ct_df[conn_ct_df['roi'].eq(av_rois)]

        
        rel_frac = rel_frac + '_by_roi'
        rel_tot = rel_tot + '_by_roi'

        if prop_to_plot == 'num_to_thresh':
            plot_data_df = self.__calculate_num_celltypes_to_frac_thresh(conn_ct_df,
                rel_bid, rel_frac, rel_tot, frac_thresh)
            rel_plot_col = 'num_types_to_thresh'
            plot_tit = f'{prop_to_plot} thresh:{str(frac_thresh)}'
            hover5 = 'tot_num_syn'
        else:
            plot_data_df = self.__calculate_num_syn_and_conn_w_thresh(conn_ct_df,
                rel_bid, min_num_syn)
            if prop_to_plot == 'syn':
                rel_plot_col = 'num_syn'
            else:
                rel_plot_col = 'num_conn'
            plot_tit = f'{prop_to_plot} min:{str(min_num_syn)}'
            hover5 = 'num_syn'

        if prop_thresh:
            bin_data = plot_data_df[rel_plot_col] >= prop_thresh
            plot_data_df.insert(len(plot_data_df.columns), 'bin_data', bin_data)
            rel_plot_col = 'bin_data'

        fig = px.scatter_3d(
            plot_data_df, x='x_post', y='y_post', z='z_post',
            color=rel_plot_col, color_continuous_scale=self.__colormap_seq,
            hover_name=rel_bid,
            hover_data= {
                'x_post':False, 'y_post':False, 'z_post':False, 
                rel_plot_col:True, hover5:True})
        fig.update_traces(marker_size = 4)

        fig.update_layout(
            scene={
                'xaxis': {'showgrid': False, 'backgroundcolor': self.__color_background},
                'yaxis': {'showgrid': False, 'backgroundcolor': self.__color_background},
                'zaxis': {'showgrid': False, 'backgroundcolor': self.__color_background}})

        self.__fig_dict[f'plot_neuron_properties_by_spatial_position ({plot_tit}):{conn_dir}'] = fig

        if show_flag:
            fig.show()


    def __calculate_num_celltypes_to_frac_thresh(self,
        con_ct_df, body_id, frac, tot, f_thresh
        ) -> pd.DataFrame:

        firstrank_df = con_ct_df[con_ct_df['rank_first'] == 1]
        firstrank_df = firstrank_df\
            .sort_values(by=[body_id, frac], ascending=[True, False])\
            .reset_index(drop=True)
        firstrank_df['cumsum_frac_tot'] = firstrank_df\
            .groupby(body_id)[frac]\
            .transform('cumsum')

        # rank is used to show how many cell types are needed (since it is already sorted)
        firstrank_df['num_types_to_thresh'] = firstrank_df\
            .groupby(body_id)['cumsum_frac_tot']\
            .rank()
        plot_df = pd.DataFrame(firstrank_df[firstrank_df['cumsum_frac_tot'] > f_thresh]\
            .groupby(body_id).first()['num_types_to_thresh'])\
            .reset_index()
        com_df = firstrank_df\
            .groupby(body_id)[['x_post', 'y_post', 'z_post', tot, 'tot_num_syn']]\
            .first()
        plot_df = plot_df.merge(com_df, on=body_id)

        return plot_df


    def __calculate_num_syn_and_conn_w_thresh(self,
        con_ct_df, body_id, min_syn
        ) -> pd.DataFrame:

        temp_df = con_ct_df[con_ct_df['syn_count'] >= min_syn]
        plot_df = temp_df\
            .groupby(body_id)[['x_post', 'y_post', 'z_post']]\
            .first()\
            .reset_index()
        plot_df['num_syn'] = temp_df\
            .groupby(body_id)['syn_count']\
            .sum()\
            .values
        plot_df['num_conn'] = temp_df\
            .groupby(body_id)['syn_count']\
            .count()\
            .values

        return plot_df


    def plot_celltype_conn_by_spatial_position(
        self
      , conn_dir: str
      , conn_celltype: str
      , prop_to_plot: str
      , roi_to_plot: str=None
      , min_num_syn: int=1
      , prop_thresh: float=None
      , show_flag: bool=True
    ):
        """
        This functon plots a connected cell type property by the  average position of its
          input(output) synapses

        Parameters
        ----------
        conn_dir : str
            'input' : when plotting all the different input celltypes.
            'output' : when plotting all the different output celltypes.
        conn_celltype : str
            cell type name for which the properties will be calculated
        prop_to_plot : str
            property to plot, can be one of 3
            'syn' : total number of synapses from the given celltype
            'frac': total fraction of input/output from the given cell type (out of all
              connection regardless of min_num_syn) for the given ROI
            'frac_tot' : fraction of input/output from the given cell type but out of all input/outputs (not just from the ROI)
            'conn' : total number of cvonnections from the given cell type
        roi_to_plot : str=None, 
            giving the name of a particular ROI will plot data only from that ROI, if there are multiple ROIs this input is neccessary
            if there is only one ROI in the object that ROI will be plotted
        min_num_syn : int, default=1
            adjusts only 'tot_syn'. minimum number of synapses to consider a connection
        prop_thresh : int, default=None
            if given turns the plot into a binary plot, with values above and below the given
              number plotted in 2 different colors
            (applies to all plotted properties)
        show_flag : bool, default=True
            if true adds the figure to the fig_dict and also plots to screen
        
        Returns
        -------
        None
        """
        ctype_cbyr = self.get_ctype_connbyroi()

        assert conn_dir in ['input', 'output'], 'conn_dir should be either input or output'
        assert prop_to_plot in ['syn', 'frac', 'conn', 'frac_tot'],\
            'prop_to_plot should be either syn, conn, frac or frac_tot'

        if conn_dir == 'input':

            conn_ct_df = ctype_cbyr.get_input_neurons_w_stats()

            rel_bid = 'bodyId_post'
            ct_type = 'type_pre'
            rel_frac = 'frac_tot_pre'
        else:

            conn_ct_df = ctype_cbyr.get_output_neurons_w_stats()

            rel_bid = 'bodyId_pre'
            ct_type = 'type_post'
            rel_frac = 'frac_tot_post'
        
        av_rois = conn_ct_df['roi'].unique()

        if (not isinstance(av_rois, str)) & (len(av_rois) > 1):  
            if roi_to_plot is None:
                raise ValueError(
                    f"an ROI must be selected from {av_rois}")
            else:
                assert roi_to_plot in av_rois, f"ROI should be from the following: {av_rois}"
                conn_ct_df = conn_ct_df[conn_ct_df['roi'].eq(roi_to_plot)]
        elif isinstance([av_rois], str):  
            conn_ct_df = conn_ct_df[conn_ct_df['roi'].eq(av_rois)]

        all_ct_names = conn_ct_df[ct_type].unique()
        assert conn_celltype in all_ct_names,\
            f"given cell type not in {ct_type} conn cells are {all_ct_names}"

        plot_data_df = self.__calculate_celltype_syn_w_thresh(
            conn_ct_df, rel_bid, ct_type, conn_celltype, min_num_syn, rel_frac)

        plot_tit = f'{prop_to_plot} min:{str(min_num_syn)}'

        if prop_thresh:
            bin_data = plot_data_df[prop_to_plot] >= prop_thresh
            plot_data_df.insert(len(plot_data_df.columns), 'bin_data', bin_data)
            prop_to_plot = 'bin_data'

        fig = px.scatter_3d(
            plot_data_df, x='x_post', y='y_post', z='z_post',
            color=prop_to_plot, color_continuous_scale=self.__colormap_seq,
            hover_name=rel_bid,
            hover_data= {'x_post':False, 'y_post':False, 'z_post':False, prop_to_plot:True})
        fig.update_traces(marker_size = 4)

        fig.update_layout(
            scene={
                'xaxis': {'showgrid': False, 'backgroundcolor': self.__color_background},
                'yaxis': {'showgrid': False, 'backgroundcolor': self.__color_background},
                'zaxis': {'showgrid': False, 'backgroundcolor': self.__color_background}})

        self.__fig_dict[f'plot_celltype_conn_by_spatial_position ({plot_tit}):{conn_dir}'] = fig

        if show_flag:
            fig.show()


    def __calculate_celltype_syn_w_thresh(self,
        con_ct_df, body_id, ct_typ, ct_nam, min_syn, rel_frc
        ) -> pd.DataFrame:
        rel_f = rel_frc + '_by_roi'
        rel_f_t = rel_frc + '_roi_tot'
        temp_df = con_ct_df[(con_ct_df['syn_count'] >= min_syn) & (con_ct_df[ct_typ] == ct_nam)]
        plot_df = temp_df\
            .groupby(body_id)[['x_post', 'y_post', 'z_post', rel_frc]]\
            .mean()\
            .reset_index()  # TODO: potentialy `first()` instead of `mean()`?
        plot_df['frac_tot'] = temp_df\
            .groupby(body_id)[rel_f_t]\
            .first()\
            .values
        plot_df['syn'] = temp_df\
            .groupby(body_id)['syn_count']\
            .sum()\
            .values
        plot_df['conn'] = temp_df\
            .groupby(body_id)['syn_count']\
            .count()\
            .values
        plot_df\
            .rename({rel_frc: 'frac'}, axis=1, inplace=True)

        return plot_df

    def plot_num_celltypes_to_threshold(
        self,
        conn_dir:str,
        frac_thresh:list=[0.5, 0.75, 0.8, 0.9],
        show_flag: bool=True
        ):
        """
        Plots histogram for the number on input/output cell types required to reach a certain
          connectivity threshold

        This function plots a histogram for each value in frac_thresh, which is the distribution of
          pre/post cell types needed to reach that fraction of inputs/outputs

        Parameters
        ----------
        conn_dir : str
            'input' : when plotting all the different input celltypes.
            'output' : when plotting all the different output celltypes.
        frac_thresh : list (of floats), default = [0.5, 0.75, 0.8, 0.9]
            list of thresholds for fraction of input or output to reach
        show_flag : bool (default=True)
            if true adds the figure to the fig_dict and also plots to screen

        Returns
        -------
        None
        """
        assert conn_dir in ['input', 'output'], 'conn_dir should be either input or output'
        ctype_cbyr = self.get_ctype_connbyroi()

        if conn_dir == 'output':
            rel_frac = 'frac_tot_post'
            rel_bid = 'bodyId_pre'
            conn_ct_df = ctype_cbyr.get_output_neurons_w_stats()
        else:
            rel_frac = 'frac_tot_pre'
            rel_bid = 'bodyId_post'
            conn_ct_df = ctype_cbyr.get_input_neurons_w_stats()

        if conn_ct_df is None:
            raise ValueError(f"no connectivity dataframe - run calc {conn_dir} neurons w stats")

        firstrank_df = conn_ct_df[conn_ct_df['rank_first'] == 1]
        firstrank_df = firstrank_df\
            .sort_values(by=[rel_bid, rel_frac], ascending=[True, False])\
            .reset_index(drop=True)
        firstrank_df['cumsum_frac_tot'] = firstrank_df\
            .groupby(rel_bid)[rel_frac]\
            .transform('cumsum')
        firstrank_df['num_types_to_thresh'] = firstrank_df\
            .groupby(rel_bid)['cumsum_frac_tot']\
            .rank() # rank is used to show how many cell types are needed (already sorted)

        # stopped using this since if the cutoff is not reached (due to None pre/post types) the
        #   cell is dropped from the calculation
        # cuttoff_array = np.empty(shape=(np.size(frac_thresh), firstrank_df[rel_bid].nunique()))

        plot_tit = ['None'] * np.size(frac_thresh)
        for ind, thresh_total in enumerate(frac_thresh):
            plot_tit[ind] = f'thresh val:{str(thresh_total)}'

        fig = make_subplots(cols=np.size(frac_thresh), rows=1, subplot_titles= plot_tit)

        for ind, thresh_total in enumerate(frac_thresh, start=1):
            temp_arry = firstrank_df\
                .query(f'cumsum_frac_tot > {str(thresh_total)}')\
                .groupby(rel_bid)\
                .first()['num_types_to_thresh']\
                .values

            fig.add_trace(
                go.Histogram(
                    x=temp_arry,
                    marker={'color': self.__color_bar},
                    showlegend=False),
                col=ind, row=1)

        self.__fig_dict[f"plot_num_celltypes_to_threshold:{conn_dir}"] = fig

        if show_flag:
            fig.show()


    def plot_distribution_connected_cells_and_syn(
        self
      , conn_dir: str
      , show_flag: bool=True
    ):
        """
        Plots a 3 general statistics plots first:
          - histogram of number of pre(post)-cells connected to each post(pre)
          - histogram of the number of total synapses for each post(pre)
          - scatter plot of the above two variables

        These are meant to estimate the homogeneity in the post(pre) population

        Parameters
        ----------
        conn_dir : str
            'input': when plotting all the different input celltypes.
            'output': when plotting all the different output celltypes.
        show_flag : bool, default=True
            if true adds the figure to the fig_dict and also plots to screen
        """
        # managing inputs
        assert conn_dir in ['input', 'output'], 'conn_dir should be either input or output'
        ctype_cbyr = self.get_ctype_connbyroi()

        if conn_dir == 'input':
            dir_type = 'pre'
            sing_type = 'post'
            rel_tot_num = 'tot_num_inp'
            conn_ct_df = ctype_cbyr.get_input_neurons_w_stats()
        else:
            dir_type = 'post'
            sing_type = 'pre'
            rel_tot_num = 'tot_num_out'
            conn_ct_df = ctype_cbyr.get_output_neurons_w_stats()

        if conn_ct_df is None:
            raise ValueError(f"no connectivity dataframe - run calc {conn_dir} neurons w stats")

        body_id_type = f'bodyId_{sing_type}'

        glob_df = conn_ct_df\
            .groupby([body_id_type])\
            .first()\
            .reset_index()
        num_bins_g = max([int(np.floor(glob_df.shape[0]/35)), 20])
        tit1 = f"# of {dir_type} cells per {sing_type}"
        tit2 = f"# of total synapses per {sing_type}"
        fig = make_subplots(rows=1, cols=3, subplot_titles=(tit1, tit2, None))

        ylab1 = f"# of {sing_type} cells"
        xlab2 = f"# of {dir_type} cells per {sing_type}"
        ylab2 = f"# of total synapses per {sing_type}"

        fig.add_trace(
            go.Histogram(
                x=glob_df[rel_tot_num],
                nbinsx=num_bins_g,
                marker={'color': self.__color_bar}),
            row=1, col=1)
        fig.add_trace(
            go.Histogram(
                x=glob_df['tot_num_syn'],
                nbinsx=num_bins_g,
                marker={'color': self.__color_bar}),
            row=1, col=2)
        fig.add_trace(
            go.Scatter(
                x=glob_df[rel_tot_num],
                y=glob_df['tot_num_syn'],
                mode='markers',
                marker={
                    'color': self.__color_bar,
                    'size': 4},
                hovertemplate='bodyID:%{text}<br>conn:%{x}<br>syn:%{y}<extra></extra>',
                # hoverinfo='text',
                text=glob_df[body_id_type].values),
            row=1, col=3)
        fig.update_layout(
            yaxis={'title_text': ylab1, 'showgrid': False},
            yaxis2={'title_text': ylab1, 'showgrid': False},
            xaxis3={'title_text': xlab2, 'showgrid': False},
            yaxis3={'title_text': ylab2, 'showgrid':False},
            showlegend=False)

        self.__fig_dict[f'plot_distribution_connected_cells_and_syn:{conn_dir}'] = fig

        if show_flag:
            fig.show()


    def plot_target_celltype_centric_stats(
        self
      , conn_dir: str
      , conn_cell_types: list # or np.ndarray
      , rank_thresh: int=4
      , common_axes: bool=True
      , show_flag: bool=True
    ):
        """
        Plots histograms and boxplots to summerize connectivity from the perspective of the target
          celltype.

        First, plots 2 histograms for each cell in conn_cell_types
            histogram of number of specific pre(post)_type per post(pre)
            histogram of number total synapses from the above sepcific pre(post)-type per post(pre)

        Next, plots 2 boxplots for each cell type in conn_cell_types up to the rank given in
          rank_thresh.
        plots are for number of pre(post) to post(pre) synapses and fraction of pre(post) synapses 
          from all the same pre(post) neurons. both are by rank (pre neuron with most connection 
          for the pre type)

        Parameters
        ----------
        conn_dir : str
            'input' : when plotting all the different input celltypes.
            'output' : when plotting all the different output celltypes.
        conn_cell_types : list | np.ndarray
            numpy.ndarray or list of connected cell type names to be plotted
        rank_thresh : int, default=4
            last rank to plot in the bottom 2 boxplots.
              (see rank_dense in `fetch_indiv_pretypes_of_posttype_v2`)
        common_axes : bool, default=True
            If true, histograms are plotted on a common x axis (same range)
        show_flag : bool, default=True
            if true adds the figure to the fig_dict and also plots to screen

        Returns
        -------
        fig : go.Figure
        """

        # managing inputs
        assert conn_dir in ['input', 'output'], 'conn_dir should be either input or output'
        ctype_cbyr = self.get_ctype_connbyroi()

        if conn_dir == 'input':
            dir_type = 'pre'
            sing_type = 'post'
            rel_frac_type = 'frac_inp_pre_type'
            conn_ct_df = ctype_cbyr.get_input_neurons_w_stats()
        else:
            dir_type = 'post'
            sing_type = 'pre'
            rel_frac_type = 'frac_out_post_type'
            conn_ct_df = ctype_cbyr.get_output_neurons_w_stats()

        if conn_ct_df is None:
            raise ValueError(f"no connectivity dataframe - run calc {conn_dir} neurons w stats")

        conn_type = f'type_{dir_type}'
        frac_type = f'frac_tot_{dir_type}'
        rel_tot_syn = f'tot_syn_per_{dir_type}'

        num_ct = len(conn_cell_types)

        # calculating rank by fraction from total connections
        crit = conn_ct_df['rank_first'] == 1
        temp_df = conn_ct_df[crit]
        med_frac_per_pre  = temp_df\
            .groupby([conn_type])[frac_type]\
            .median()

        rel_lab_ser = med_frac_per_pre[conn_cell_types]
        rel_lab = []
        for key, val in rel_lab_ser.items():
            rel_lab.append(f'{key} {val:.2f}')
        # sorts the list so that it will match the one after groupby and hist are applied
        rel_lab.sort()
        rel_dict = [None] * num_ct
        rel_dict[0] = {"colspan": num_ct}
        tit1 = tuple((f'# of {ct} conn') for ct in conn_cell_types)
        tit2 = tuple((f'{ct} # of syn') for ct in conn_cell_types)
        tit3 = tuple([f'# of syn from {dir_type} cells'])
        tit4 = tuple([f'fraction of synapses out of total by {dir_type}_type'])
        subplots_tit = tit1 + tit2 + tit3 + tit4
        fig = make_subplots(rows=4, cols=num_ct,
            specs=[[{}] * num_ct, [{}] * num_ct, rel_dict, rel_dict],
            subplot_titles=subplots_tit
            )

        col_name = f"num_{dir_type}_per_{sing_type}"

        plot_df = conn_ct_df[(conn_ct_df[conn_type].isin(conn_cell_types)) & crit].copy()

        if common_axes:
            max_cells = plot_df[col_name].max()
            binx = go.histogram.XBins(start=0.5, end=max_cells+0.5)

        for ind, cell_type in enumerate(conn_cell_types, start=1):
            if common_axes:
                fig.add_trace(
                    go.Histogram(
                        x=plot_df[plot_df[conn_type] == cell_type][col_name]
                      , xbins=binx
                      , marker={'color': self.__color_bar}
                    )
                  , row=1, col=ind
                )
            else:
                fig.add_trace(
                    go.Histogram(
                        x=plot_df[plot_df[conn_type] == cell_type][col_name]
                      , marker={'color': self.__color_bar}
                    )
                  , row=1, col=ind
                )

        x_mins = []
        x_maxs = []
        for t_data in fig.data[0:num_ct]:
            x_mins.append(min(t_data.x))
            x_maxs.append(max(t_data.x))
        x_min = min(x_mins)
        x_max = max(x_maxs)

        if common_axes:
            for ind in range(num_ct):
                fig.update_xaxes(range=[x_min, x_max], row=1, col=ind+1)

        if common_axes:
            max_cells = plot_df[rel_tot_syn].max()
            binx = go.histogram.XBins(start=0.5, end=max_cells+0.5)

        for ind, cell_type in enumerate(conn_cell_types, start=1):
            if common_axes:
                fig.add_trace(
                    go.Histogram(
                        x=plot_df[plot_df[conn_type] == cell_type][rel_tot_syn]
                      , xbins=binx
                      , marker={'color': self.__color_bar}
                    )
                  , row=2, col=ind
                )
            else:
                fig.add_trace(
                    go.Histogram(
                        x=plot_df[plot_df[conn_type] == cell_type][rel_tot_syn]
                      , marker={'color': self.__color_bar}
                    )
                  , row=2, col=ind
                )

        x_mins = []
        x_maxs = []
        for t_data in fig.data[num_ct:2*num_ct]:
            x_mins.append(min(t_data.x))
            x_maxs.append(max(t_data.x))
        x_min = min(x_mins)
        x_max = max(x_maxs)
        if common_axes:
            for ind in range(num_ct):
                fig.update_xaxes(range=[x_min, x_max], row=2, col=ind+1)

        # plotting by ranks
        crit = np.logical_and(\
            conn_ct_df[conn_type].isin(conn_cell_types),\
            conn_ct_df['rank_dense'] <= rank_thresh)
        rank_df = conn_ct_df[crit]

        p_cols = pc.get_colorscale(self.__colormap_seq)

        # boxplot_x_vals = rank_df[conn_type]
        box_offset = range(rank_thresh)

        for ind in range(rank_thresh):
            temp_df = rank_df[rank_df['rank_dense'] == ind+1]
            fig.add_trace(
                go.Box(
                    x=temp_df[conn_type]
                  , y=temp_df['syn_count']
                  , marker_color= p_cols[ind+2][1]
                  , offsetgroup=box_offset[ind]
                #   , boxpoints='all'
                #   , pointpos=-1
                #   , marker_size=3
                #   , jitter=0.5
                )
              , row=3, col=1)

        for ind in range(rank_thresh):
            temp_df = rank_df[rank_df['rank_dense'] == ind+1]
            fig.add_trace(
                go.Box(
                    x=temp_df[conn_type]
                  , y=temp_df[rel_frac_type]
                  , marker_color= p_cols[ind+2][1]
                  , offsetgroup=box_offset[ind]
                #   , boxpoints='all'
                #   , pointpos=-1
                #   , marker_size=3
                #   , jitter=0.5
                )
              , row=4, col=1)

        fig_wid = max(600, 200 * num_ct)

        fig.update_layout(
            boxmode='group'
          , autosize=False
          , width=fig_wid
          , height=1200
          , showlegend=False
        )
        ct_names = ' '.join([str(elem) for elem in conn_cell_types])
        self.__fig_dict[
            f'plot_target_celltype_centric_conn:{conn_dir}_celltypes:{ct_names}'] = fig

        if show_flag:
            fig.show()


    def plot_connected_celltype_centric_stats(
        self,
        conn_dir: str,
        conn_cell_types: list, # or np.ndarray
        rank_thresh: int=4, common_axes: bool=True,
        show_flag: bool=True
    ):
        """
        Plots histograms and boxplots to summerize connectivity from the perspective of the
        connected celltypes.

        First, plots 2 histograms for each cell in conn_cell_types
            histogram of number of specific pre(post)_type per post(pre)
            histogram of number total synapses from the above sepcific pre(post)-type per post(pre)

        Next, plots 2 boxplots for each cell type in conn_cell_types up to the rank given in
          rank_thresh.
        plots are for number of pre(post) to post(pre) synapses and fraction of pre(post) synapses
          from all the same pre(post) neurons
        both are by rank (pre neuron with most connection for the pre type)

        Parameters
        ----------
        conn_dir : str
            'input' : when plotting all the different input celltypes.
            'output' : when plotting all the different output celltypes.
        conn_cell_types : list
            numpy.ndarray or list of connected cell type names to be plotted
        rank_thresh : int, default=4
            last rank to plot in the bottom 2 boxplots.
            (see rank_dense in fetch_indiv_pretypes_of_posttype_v2)
        common_axes : bool, default=True
            If true, histograms are plotted on a common x axis (same range)
        show_flag : bool, default=True
            if true adds the figure to the fig_dict and also plots to screen

        Returns
        -------
        fig : go.Figure
        """

        # managing inputs
        assert conn_dir in ['input', 'output'], 'conn_dir should be either input or output'
        ctype_cbyr = self.get_ctype_connbyroi()

        if conn_dir == 'input':
            dir_type = 'pre'
            sing_type = 'post'
            conn_ct_df = ctype_cbyr.get_input_neurons_w_stats()
        else:
            dir_type = 'post'
            sing_type = 'pre'
            conn_ct_df = ctype_cbyr.get_output_neurons_w_stats()

        if conn_ct_df is None:
            raise ValueError(f"no connectivity dataframe - run calc {conn_dir} neurons w stats")

        conn_type = f'type_{dir_type}'
        rel_frac_syn_stot = f'frac_syn_tot_{sing_type}'
        rel_snum = f'num_{sing_type}'
        rel_tot_ssyn = f'tot_syn_{sing_type}'
        frac_type = f'frac_tot_{dir_type}'

        num_ct = len(conn_cell_types)

        # calculating rank by fraction from total connections
        crit = conn_ct_df['rank_first'] == 1
        temp_df = conn_ct_df[crit]
        med_frac_per_pre  = temp_df\
            .groupby([conn_type])[frac_type]\
            .median()

        rel_lab_ser = med_frac_per_pre[conn_cell_types]
        rel_lab = []
        for key, val in rel_lab_ser.items():
            rel_lab.append(f'{key} {val:.2f}')
        # sorts the list so that it will match the one after groupby and hist are applied
        rel_lab.sort()

        pre_df = self.__calc_connected_celltypes_df(conn_ct_df, dir_type)

        crit = np.logical_and(\
        pre_df[conn_type].isin(conn_cell_types), \
        pre_df['count_rank_dense'] ==1)
        plot_df = pre_df[crit]

        rel_dict = [None] * num_ct
        rel_dict[0] = {"colspan": num_ct}
        tit1 = tuple((f'# of target ct <br> conn to {ct}') for ct in conn_cell_types)
        tit2 = tuple((f'{ct} # of syn <br> to all targets') for ct in conn_cell_types)
        tit3 = tuple(['# of conn celltype syn'])
        tit4 = tuple([
            'fraction of synapses out of total <br> (to target type only) by conn celltype'])
        subplots_tit = tit1 + tit2 + tit3 + tit4
        fig = make_subplots(rows=4, cols=num_ct,
            specs=[[{}] * num_ct, [{}] * num_ct, rel_dict, rel_dict],
            subplot_titles=subplots_tit
        )

        if common_axes:
            max_cells = plot_df[rel_snum].max()
            binx = go.histogram.XBins(start=0.5, end=max_cells+0.5)

        for ind, cell_type in enumerate(conn_cell_types, start=1):
            if common_axes:
                fig.add_trace(
                    go.Histogram(
                        x=plot_df[plot_df[conn_type] == cell_type][rel_snum],
                        xbins=binx,
                        marker={'color': self.__color_bar}),
                    row=1, col=ind)
            else:
                fig.add_trace(
                    go.Histogram(
                        x=plot_df[plot_df[conn_type] == cell_type][rel_snum],
                        marker={'color':self.__color_bar}),
                    row=1, col=ind)

        x_mins = []
        x_maxs = []
        for t_data in fig.data[0:num_ct]:
            x_mins.append(min(t_data.x))
            x_maxs.append(max(t_data.x))
        x_min = min(x_mins)
        x_max = max(x_maxs)

        if common_axes:
            for ind in range(num_ct):
                fig.update_xaxes(range=[x_min, x_max], row=1, col=ind+1)


        if common_axes:
            max_cells = plot_df[rel_tot_ssyn].max()
            binx = go.histogram.XBins(start=0.5, end=max_cells+0.5)

        for ind, cell_type in enumerate(conn_cell_types, start=1):
            if common_axes:
                fig.add_trace(
                    go.Histogram(
                        x=plot_df[plot_df[conn_type] == cell_type][rel_tot_ssyn],
                        xbins=binx,
                        marker={'color': self.__color_bar}),
                    row=2, col=ind)
            else:
                fig.add_trace(
                    go.Histogram(
                        x=plot_df[plot_df[conn_type] == cell_type][rel_tot_ssyn],
                        marker={'color': self.__color_bar}),
                    row=2, col=ind)

        x_mins = []
        x_maxs = []
        for t_data in fig.data[num_ct:2*num_ct]:
            x_mins.append(min(t_data.x))
            x_maxs.append(max(t_data.x))
        x_min = min(x_mins)
        x_max = max(x_maxs)
        if common_axes:
            for ind in range(num_ct):
                fig.update_xaxes(range=[x_min, x_max], row=2, col=ind+1)

        # plotting by ranks
        crit2 = np.logical_and(\
            pre_df[conn_type].isin(conn_cell_types),\
            pre_df['count_rank_dense'] <= rank_thresh)
        rank_df = pre_df[crit2]

        p_cols = pc.get_colorscale(self.__colormap_seq)

        box_offset = range(rank_thresh)

        for ind in range(rank_thresh):
            temp_df = rank_df[rank_df['count_rank_dense'] == ind+1]
            fig.add_trace(
                go.Box(
                    x=temp_df[conn_type],
                    y=temp_df['syn_count'],
                    marker_color= p_cols[ind+2][1],
                    offsetgroup=box_offset[ind]
                    # boxpoints='all',
                    # pointpos=-1,
                    # marker_size=3,
                    # jitter=0.5
                ),
            row=3, col=1)

        for ind in range(rank_thresh):
            temp_df = rank_df[rank_df['count_rank_dense'] == ind+1]
            fig.add_trace(
                go.Box(
                    x=temp_df[conn_type],
                    y=temp_df[rel_frac_syn_stot],
                    marker_color= p_cols[ind+2][1],
                    offsetgroup=box_offset[ind]
                    # boxpoints='all',
                    # pointpos=-1,
                    # marker_size=3,
                    # jitter=0.5
                ),
            row=4, col=1)

        fig_wid = max(600, 200 * num_ct)
        fig.update_layout(
            boxmode='group',
            autosize=False,
            width=fig_wid,
            height=1200,
            showlegend=False
            )
        
        ct_names = ' '.join([str(elem) for elem in conn_cell_types])
        self.__fig_dict[
            f'plot_connected_celltype_centric_stats:{conn_dir}_celltypes:{ct_names}'] = fig

        if show_flag:
            fig.show()


    def __calc_connected_celltypes_df(
        self,
        conn_ct_df: pd.DataFrame,
        dir_type: str
    ) -> pd.DataFrame:
        """
        internal function for processing the dataframe in plot_projective_field
        """
        conn_type = f'type_{dir_type}'
        if dir_type == 'pre':
            sing_type = 'post'
        else:
            sing_type = 'pre'

        body_id_stype = f'bodyId_{sing_type}'  # single
        body_id_ptype = f'bodyId_{dir_type}'  # plural
        rel_snum = f'num_{sing_type}'
        rel_tot_ssyn = f'tot_syn_{sing_type}'
        rel_frac_syn_stot = f'frac_syn_tot_{sing_type}'

        # cutting the relevant input df into a pretype centric one
        #   (most of the stats are post centric)
        pre_df = conn_ct_df[[body_id_stype, body_id_ptype, conn_type, 'syn_count']]\
            .reset_index(drop=True)

        # using transform generates a column the same size as the original df (by repeating values)
        pre_df[rel_snum] = pre_df\
            .groupby([conn_type, body_id_ptype])['syn_count']\
            .transform('count')
        pre_df[rel_tot_ssyn] = pre_df\
            .groupby([conn_type, body_id_ptype])['syn_count']\
            .transform('sum')
        pre_df = pre_df\
            .sort_values(by=[conn_type, body_id_ptype], ascending=[True, False])
        pre_df[rel_frac_syn_stot] = pre_df['syn_count']\
            .div(pre_df[rel_tot_ssyn].values)
        pre_df['count_rank_dense'] = pre_df\
            .groupby([conn_type, body_id_ptype])['syn_count']\
            .rank('dense', ascending=False)

        return pre_df


    def plot_input_output_celltype_consistency(
            self,
            roi_to_plot: str=None,
            prop_to_plot: str='frac',
            log_flag: bool=False,
            show_flag: bool=True
        ) -> pd.DataFrame :
        """
        Plots scatter plot of input and output stats by celltype to the target celltype.

        Parameters
        ----------
        roi_to_plot : str, default=None
            giving the name of a particular ROI will plot data only from that ROI, if not given plots all data

        prop_to_plot : str
            propretry to plot, can be one of 3
            'syn' : plots the median number of total synapses from connected celltype to target
              celltype vs. median number of total synapses from target celltype to the same
              connected celltype
            'conn' : same as above for the median number of total connections (number of cells)
            'frac' : plots the fraction a connected celltype is from all the inputs to the
              target celltype vs. the fraction the same connected celltype is from all the output
              of the target celltype

        log_flag : bool, default=False
            if True plots both axes on a log plot

        show_flag : bool, default=True
            if true adds the figure to the fig_dict and also plots to screen

        Returns
        -------
        return_df : pd.DataFrame
            dataframe of the combined input and output celltype stats sorted by the plotted
              value
        """
        assert prop_to_plot in ['syn', 'conn', 'frac'],\
            'prop_to_plot should be either syn, conn or frac'
        
        # managing inputs
        ctype_obj = self.get_ctype_connbyroi()
        target_inp_df = ctype_obj.get_input_celltypes_w_stats()
        target_out_df = ctype_obj.get_output_celltypes_w_stats()
        if target_inp_df is None:
            raise ValueError("input_celltypes_w_stats is empty - run calc input celltypes w stats")
        if target_out_df is None:
            raise ValueError("output_celltypes_w_stats is empty - run calc out celltypes w stats")
        
        av_rois = set(target_out_df['roi'].unique()).union(set(target_inp_df['roi'].unique()))
        if roi_to_plot is not None:
            
            assert roi_to_plot in av_rois,\
                f"ROI should be from the following: {av_rois} or None for ALL"
            target_inp_df = target_inp_df[target_inp_df['roi'].eq(roi_to_plot)]
            target_out_df = target_out_df[target_out_df['roi'].eq(roi_to_plot)]
            match prop_to_plot:
                case 'syn':
                    target_inp_df.rename(columns = {'tot_syn_per_preNroi':'syn'}, inplace=True)
                    target_out_df.rename(columns = {'tot_syn_per_postNroi':'syn'}, inplace=True)
                case 'conn':
                    target_inp_df.rename(columns = {'tot_conn_per_preNroi':'conn'}, inplace=True)
                    target_out_df.rename(columns = {'tot_conn_per_postNroi':'conn'}, inplace=True)
                case 'frac':    
                    target_inp_df.rename(columns = {'frac_tot_pre_roi_tot':'frac'}, inplace=True)
                    target_out_df.rename(columns = {'frac_tot_post_roi_tot':'frac'}, inplace=True)
        else: # plots all ROIs together
            if prop_to_plot == 'frac':
                target_inp_df = target_inp_df\
                    .groupby('type_pre', as_index=False)['frac_tot_pre_roi_tot']\
                    .sum()\
                    .rename(columns = {'frac_tot_pre_roi_tot': 'frac'}) 
                target_out_df= target_out_df\
                    .groupby('type_post', as_index=False)['frac_tot_post_roi_tot']\
                    .sum()\
                    .rename(columns = {'frac_tot_post_roi_tot': 'frac'}) 
            else:# these 2 already claculated for total
                target_inp_df = target_inp_df\
                    .groupby('type_pre', as_index=False)[['med_tot_syn', 'med_tot_conn']]\
                    .first()\
                    .rename(columns= {'med_tot_syn':'syn', 'med_tot_conn': 'conn'})
                target_out_df = target_out_df\
                    .groupby('type_post', as_index=False)[['med_tot_syn', 'med_tot_conn']]\
                    .first()\
                    .rename(columns= {'med_tot_syn':'syn', 'med_tot_conn': 'conn'})

        comb_df = target_inp_df\
            .merge(
                target_out_df,
                how='outer', left_on='type_pre', right_on='type_post',
                suffixes=('_pre', '_post'))
        # getting rid of NaNs to plot even celltypes with empty enteries in either input or output
        comb_df = comb_df\
            .fillna(0)

        return_df = comb_df[['type_pre', f'{prop_to_plot}_pre', f'{prop_to_plot}_post']]\
            .rename(columns={'type_pre':'conn_type'})
        return_df[f'{prop_to_plot}_comb'] = \
            return_df[[ f'{prop_to_plot}_pre', f'{prop_to_plot}_post']]\
                .pow(2)\
                .sum(axis=1)\
                .pow(1/2)
        return_df\
            .sort_values(
                by=[f'{prop_to_plot}_pre', f'{prop_to_plot}_post'],
                ascending=[False, False],
                inplace=True,
                ignore_index=True)
        
        max_x = comb_df[f'{prop_to_plot}_pre'].max()
        max_y = comb_df[f'{prop_to_plot}_post'].max()
        tot_max = np.maximum(max_x, max_y)
        tot_range = np.ceil(tot_max + tot_max/5)
        if log_flag:
            tot_range = np.log10(tot_range)


        fig = go.Figure(data=go.Scatter(
            x=comb_df[f'{prop_to_plot}_pre'],
            y=comb_df[f'{prop_to_plot}_post'],
            mode='markers+text',
            marker={
                'size': 10,
                'color':self.__color_bar
            },
            text=comb_df['type_pre'],
            textposition='top center'))
        fig.update_layout(title='Pre-Post connections', 
                          yaxis_range=[-np.ceil(tot_range/10), tot_range],
                          xaxis_range=[-np.ceil(tot_range/10), tot_range],
                          width=1000,
                          height=1000
                          )
        fig.update_xaxes(
            title_text = f"Pre-{prop_to_plot}", 
            ticks="outside", tickwidth=3, ticklen=10,
            minor=dict(ticklen=5, tickcolor="lightgray", tickwidth=1)
        )
        fig.update_yaxes(
            title_text = f"Post-{prop_to_plot}", 
            ticks="outside", tickwidth=3, ticklen=10,
            minor=dict(ticklen=5, tickcolor="lightgray", tickwidth=1)
        )

        if log_flag:
            fig.update_xaxes(type='log')
            fig.update_yaxes(type='log')

        fig.update_scenes(aspectmode='data')

        self.__fig_dict[f"plot_input_output_celltype_consitency:{prop_to_plot}"] = fig
        if show_flag:
            fig.show()

        return return_df


    def plot_input_output_consistency_by_conn_celltype(
            self
          , conn_cell_types: list
          , prop_to_plot: str='frac'
          , log_flag: bool=False
          , show_flag: bool=True
        ):
        """
        Plots scatter plot of input and output stats by celltype to the target celltype.

        Parameters
        ----------
        conn_cell_types : list
            list of names for the connected cell types to plot

        prop_to_plot : str
            propretry to plot, can be one of 3
            'syn' : plots the median number of total synapses from connected celltype to target
              celltype vs. median number of total synapses from target celltype to the same
              connected celltype
            'conn' : same as above for the median number of total connections (number of cells)
            'frac' : plots the fraction a connected celltype is from all the inputs to the target
              celltype vs. the fraction the same connected celltype is from all the output of the
              target celltype
        log_flag: bool = False
            if True plots both axes on a log plot
        show_flag: bool, default=True
            if true adds the figure to the fig_dict and also plots to screen

        Returns
        -------
        None
        """
        if prop_to_plot == 'frac':
            plot_col_pre = 'frac_tot_pre'
            plot_col_post = 'frac_tot_post'
        elif prop_to_plot == 'syn':
            plot_col_pre = 'tot_syn_per_pre'
            plot_col_post = 'tot_syn_per_post'
        elif prop_to_plot == 'conn':
            plot_col_pre = 'num_pre_per_post'
            plot_col_post = 'num_post_per_pre'
        else:
            raise ValueError("prop_to_plot should be either frac, syn or conn")
        # managing inputs
        ctype_obj = self.get_ctype_connbyroi()
        target_inp_df = ctype_obj.get_input_neurons_w_stats()
        target_out_df = ctype_obj.get_output_neurons_w_stats()
        if target_inp_df is None:
            raise ValueError("input_neurons_w_stats is empty - run calc input neurons w stats")
        if target_out_df is None:
            raise ValueError("output_neurons_w_stats is empty - run calc out neurons w stats")

        tar_inp_red = target_inp_df[target_inp_df['type_pre']\
            .isin(conn_cell_types)]\
            .groupby(['bodyId_post', 'type_pre'], as_index=False)\
            .first()
        tar_out_red = target_out_df[target_out_df['type_post']\
            .isin(conn_cell_types)]\
            .groupby(['bodyId_pre', 'type_post'], as_index=False)\
            .first()

        comb_df = tar_inp_red\
            .merge(tar_out_red,
                left_on=['bodyId_post', 'type_pre'],
                right_on=['bodyId_pre', 'type_post'],
                suffixes=['_i', '_o'])

        # getting rid of NaNs to plot even celltypes with empty enteries in either input or output
        comb_df = comb_df\
            .fillna(0)
        comb_df['conn_type_num'] = comb_df['type_pre_i']\
            .replace(conn_cell_types, range(len(conn_cell_types)))
        # generating the right strings for hovering

        max_x = comb_df[plot_col_pre].max()
        max_y = comb_df[plot_col_post].max()
        tot_max = np.maximum(max_x, max_y)
        tot_range = np.ceil(tot_max + tot_max/5)
        if log_flag:
            tot_range = np.log10(tot_range)

        type_str = comb_df['type_pre_i'].values
        body_ids = comb_df['bodyId_post_i'].values

        body_id_str = [f"{bid:.0f}" for bid in body_ids]
        point_txt = []
        for tmp_type, body_id in zip(type_str, body_id_str):
            point_txt.append(f"{tmp_type}<br>bId:{body_id}")

        scat = go.Scatter(x=comb_df[plot_col_pre],
                                    y=comb_df[plot_col_post],
                                    mode='markers',
                                    marker={
                                        'size': 10,
                                        'color': comb_df['conn_type_num'],
                                        'colorscale': 'viridis'
                                    },

                    hovertext = point_txt )
        # scat.on_hover(self.hover_fn)
        fig = go.Figure(data=scat)
        fig.update_layout(title='Pre-Post connections', 
                          yaxis_range=[-np.ceil(tot_range/10), tot_range],
                          xaxis_range=[-np.ceil(tot_range/10), tot_range],
                          width=1000,
                          height=1000
                          )
        fig.update_xaxes(
            title_text = f"{plot_col_pre}", 
            ticks="outside", tickwidth=3, ticklen=10,
            minor=dict(ticklen=5, tickcolor="lightgray", tickwidth=1)
        )
        fig.update_yaxes(
            title_text = f"{plot_col_post}", 
            ticks="outside", tickwidth=3, ticklen=10,
            minor=dict(ticklen=5, tickcolor="lightgray", tickwidth=1)
        )

        if log_flag:
            fig.update_xaxes(type='log')
            fig.update_yaxes(type='log')

        fig.update_scenes(aspectmode='data')

        self.__fig_dict[f"plot_input_output_consistency_by_conn_celltype:{prop_to_plot}"] = fig
        if show_flag:
            fig.show()

    def plot_distribution_conn_celltype_stats(
        self,
        conn_dir:str,
        conn_thresh:int=1,
        log_vals:bool=True,
        conn_celltype:str=None,
        show_flag:bool=True
        ) -> pd.DataFrame:
        """
        Plots histograms of different statistic for celltype connectivity.
        This functions uses the neurons_w_stats dataframe to claculate basic stats
        on connected cell types and plot their histogram. All the stats are calculated
        by individual target neurons and by connected celltype. For example, max_syn shows
        the maximal number of synapses an individual target cell type recieves from an
        individual connected cell type (for all cell types)

        Parameters
        ----------
        conn_dir : str
            'input' : when plotting stats for all the input celltypes.
            'output' : when plotting stats for all the  output celltypes.
        conn_thresh : int, default=1
            connections will be counted only if they are equal or greater to threshold
            Note! for fraction claculation, total connections are counted and only then
            connections weaker than threshold are removed
        log_vals : bool, default=True
            if true plots the log10 of the calculated values
        conn_celltype : str, default=None
            if a celltype is given, its stats are plotted overlaid on top of the general histogram
        show_flag : bool, default=True
            if true adds the figure to the fig_dict and also plots to screen

        Returns
        -------
        plot_df : pd.DataFrame
            Dataframe used for plotting which includes the detailed statistics
        """
        # managing inputs
        assert conn_dir in ['input', 'output'], 'conn_dir should be either input or output'
        ctype_cbyr = self.get_ctype_connbyroi()

        if conn_dir == 'input':
            b_id_type = 'bodyId_post'
            conn_type = 'type_pre'
            conn_ct_df = ctype_cbyr.get_input_neurons_w_stats()
        else:
            b_id_type = 'bodyId_pre'
            conn_type = 'type_post'
            conn_ct_df = ctype_cbyr.get_output_neurons_w_stats()

        if conn_ct_df is None:
            raise ValueError(f"no connectivity dataframe - run calc {conn_dir} neurons w stats")

        rel_prop = 'syn_count'
        tot_syn_df = conn_ct_df\
            .groupby(b_id_type, as_index=False)[rel_prop]\
            .sum()\
            .rename(columns={rel_prop: 'tot_syn_count'})
        conn_ct_df = conn_ct_df[conn_ct_df[rel_prop] >= conn_thresh]
        celltype_stat_df = conn_ct_df\
            .groupby([b_id_type, conn_type], as_index=False)\
                .agg(
                    num_conn = (rel_prop, np.count_nonzero)
                  , mean_syn = (rel_prop, 'mean')
                  , max_syn = (rel_prop, 'max')
                  , med_syn = (rel_prop, 'median')
                  , sum_syn = (rel_prop, 'sum')
                )

        plot_df = celltype_stat_df.merge(tot_syn_df, 'left', on=b_id_type)
        plot_df['tot_frac'] = plot_df['sum_syn'].div(plot_df['tot_syn_count'])
        ct_plot_df = pd.DataFrame()
        if conn_celltype is not None:
            ct_plot_df = plot_df[plot_df[conn_type] == conn_celltype]

        props_to_plot = ['sum_syn', 'med_syn', 'max_syn', 'num_conn', 'tot_frac']
        fig = make_subplots(
            rows=1, cols=len(props_to_plot),
            subplot_titles=props_to_plot
        )
        for ind, prop in enumerate(props_to_plot, start=1):
            temp_x = plot_df[prop]
            if log_vals:
                temp_x = np.log10(temp_x)
            fig.add_trace(
                        go.Histogram(
                            x=temp_x,
                            marker={'color': self.__color_bar},
                            bingroup=ind,
                            hoverinfo='none',
                            showlegend=False),
                        row=1, col=ind)
            if not ct_plot_df.empty:
                temp_ctx = ct_plot_df[prop]
                if log_vals:
                    temp_ctx = np.log10(temp_ctx)
                fig.add_trace(
                        go.Histogram(
                            x=temp_ctx,
                            marker={'color': self.__color_bar2},
                            bingroup=ind,
                            hoverinfo='none',
                            showlegend=False),
                        row=1, col=ind)
        if not ct_plot_df.empty:
            fig.update_layout(barmode='overlay')
            fig.update_traces(opacity=0.65)
        self.__fig_dict[f"plot_distribution_conn_celltype_stats:{conn_dir}"] = fig
        if show_flag:
            fig.show()

        return plot_df
