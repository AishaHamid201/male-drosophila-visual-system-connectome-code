# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
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
from utils.ol_types import OLTypes

c = olc_client.connect(verbose=True)

# %%
olt = OLTypes()
oli_list = olt.get_neuron_list()

# %%
iter_counter = 0
for idx, row in oli_list.reset_index().sample(frac=1).iterrows():
    iter_counter += 1
    all_cell_dict = {}
    txt_pos = 0.92

    body_id_dict = {}

    a_bag = NeuronBag(cell_type=row['type'])
    body_id = a_bag.get_body_ids(1)[0]
    if isinstance(row['star_neuron'], int):
        body_id = row['star_neuron']

    camera_dict = get_rend_params('camera', 'whole_brain')
    scalebar_dict = get_rend_params('scalebar', 'whole_brain')

    group_dict = {}
    body_id_dict = {
        'type': row['type']
      , 'body_ids': [body_id]
      , 'body_color': [0,0,0,1]
      , 'text_position': [0.03, txt_pos]
      , 'text_align': 'l'
      , 'number_of_cells': a_bag.size
      , 'slice_width': 0
    }

    group_dict[row['type']] = body_id_dict

    generate_gallery_json(
        type_of_plot="Full-Brain"
      , description="star_wb"
      , type_or_group=row['type']
      , title=""
      , view='whole_brain'
      , list_of_ids=group_dict
      , neuropil_color=[]
      , camera=camera_dict
      , slicer={}
      , scalebar=scalebar_dict
      , n_vis={}
      , directory='all_star_gallery'
      , template="gallery-descriptions.json.jinja"
    )
    print(f"Json generation done for {row['type']}")

    # Stop if number of iterations exceed the environment variable `GALLERY_EXAMPLES`
    stop_after = os.environ.get('GALLERY_EXAMPLES')
    if stop_after:
        if stop_after := int(stop_after):
            if stop_after <= iter_counter:
                break
        else:
            break

# %%
