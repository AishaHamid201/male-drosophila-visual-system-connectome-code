# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: default
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Fill and save the template

# %%
import sys
import os
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
print(f"Project root directory: {PROJECT_ROOT}")

from utils.neuron_bag import NeuronBag
from utils.gallery_filler import generate_gallery_json
from utils.rend_params import get_rend_params
from utils import olc_client
from utils.ol_color import OL_COLOR
from utils.ol_types import OLTypes

c = olc_client.connect(verbose=True)

# %%
olt = OLTypes()
groups_by_types = olt.get_neuron_list(primary_classification=['VPN', 'VCN', 'other'])

# %%
"""
Generate JSON description files for VPN, VCN, and other groups 
using "gallery-descriptions.json.jinja" template
"""
neuropil_color = []
color_list = OL_COLOR.OL_TYPES.rgba \
    + OL_COLOR.OL_DARK_TYPES.rgba \
    + OL_COLOR.OL_LIGHT_TYPES.rgba
order = [0, 6, 2, 13, 4, 5, 1, 7, 3, 9, 12, 10]
color_list[:] = [color_list[idx] for idx in order]

iter_counter = 0
for group_name, n_types_in_group in groups_by_types.groupby('figure_group'):
    iter_counter += 1
    group_dict = {}
    txt_pos_y = 0.92

    the_view = 'whole_brain'
    txt_pos_x = 0.97
    txt_align = 'r'
    if 'VPN' in  n_types_in_group['main_groups'].tolist() \
        and 'whole_brain' not in n_types_in_group['fb_view'].tolist():
        the_view = 'half_brain'
        txt_pos_x = 0.03
        txt_align = 'l'

    for idx, row in n_types_in_group\
      .sort_values(['type'], key=lambda col: col.str.lower())\
      .reset_index()\
      .iterrows():
        txt_pos_y = txt_pos_y - 0.06

        match row['main_groups']:
            case 'VPN':
                the_directory = 'vpn_group_plots'
            case 'VCN':
                the_directory = 'vcn_group_plots'
            case _:
                the_directory = 'other_neuron_group_plots'

        camera_dict = get_rend_params('camera', the_view)
        scalebar_dict = get_rend_params('scalebar', the_view)

        a_bag = NeuronBag(cell_type=row['type'], side='R-dominant')

        sorted_body_ids = a_bag.get_body_ids(a_bag.size)
        body_id_list = sorted_body_ids.tolist()

        body_id_dict = {
            'type': row['type']
          , 'body_ids': body_id_list
          , 'body_color': color_list[idx % len(color_list)]
          , 'text_position': [txt_pos_x, txt_pos_y]
          , 'text_align': txt_align
          , 'number_of_cells': len(sorted_body_ids)
          , 'slice_width': 0
        }

        group_dict[row['type']] = body_id_dict

        generate_gallery_json(
            type_of_plot="Full-Brain"
          , description="Group"
          , type_or_group=row['figure_group']
          , title=row['figure_group'].replace("_"," ")
          , view=the_view
          , list_of_ids=group_dict
          , neuropil_color=[]
          , camera=camera_dict
          , slicer={}
          , scalebar=scalebar_dict
          , n_vis={}
          , directory=the_directory
          , template="gallery-descriptions.json.jinja"
        )
    print(f"Json generation done for {group_name}")

    # Stop if number of iterations exceed the environment variable `GALLERY_EXAMPLES`
    stop_after = os.environ.get('GALLERY_EXAMPLES')
    if stop_after:
        if stop_after := int(stop_after):
            if stop_after <= iter_counter:
                break
        else:
            break

# %%
