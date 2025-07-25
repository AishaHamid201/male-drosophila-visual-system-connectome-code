"""
Helper script to identify lists for gallery / movie / summary figures.
"""

from pathlib import Path
import pandas as pd
from dotenv import find_dotenv

from utils.ol_color import OL_COLOR

def get_one_off_params():
    """
    wrapper function to get a list of parameters in a dictionary.

    Returns:
    --------
    plots : dict
        Plotting parameters for one-off figures.
    """
    plots = {
        "Fig2a_fischbach_style": {
            'columnar_list': [
                'Mi1', 'Mi4', 'Mi9', 'C2', 'C3', 'L1', 'L2', 'L3', 'L5', 'T1'
              , 'Tm1', 'Tm2', 'Tm4', 'Tm9', 'Tm20'
            ]
          , 'list_bids_to_plot': []
          , 'hex_assign': [24, 12, -1, 1]
          , 'text_placement': [0.78, 0.95, 0,-0.06]
          , 'replace': {'name': 'Mi1', 'bid': [38620]}
          , 'directory': "fig_2"
          , 'body_color_list':
                OL_COLOR.OL_TYPES.rgba + OL_COLOR.OL_DARK_TYPES.rgba
              + OL_COLOR.OL_LIGHT_TYPES.rgba
          , 'body_color_order': [0, 6, 2, 13, 4, 5, 1, 7, 3, 9, 12, 10]
          , 'color_by':'type'
          , 'n_vis':{}
          , 'view': 'Equator_slice'
        }
      , "Fig2b_Mi9_Mi4": {
            'columnar_list': ['Mi9', 'Mi4']
          , 'list_bids_to_plot':[]
          , 'hex_assign': [17, 19,-1, 1]
          , 'text_placement': [0.78, 0.95, 0,-0.06]
          , 'replace': {}
          , 'directory': "fig_2"
          , 'body_color_list': OL_COLOR.MAGENTA_AND_GREEN.rgba
          , 'body_color_order': [1, 0]
          , 'color_by':'type'
          , 'n_vis': {}
          , 'view': 'Equator_slice'
        }
      , "FigED5e_Tm5a_Tm5b_Tm29": {
            'columnar_list': ['Tm5a', 'Tm5b', 'Tm29']
          , 'list_bids_to_plot': [557255, 132379, 69794]
          , 'hex_assign': []
          , 'text_placement':[0.78, 0.95, 0,-0.06]
          , 'replace': {}
          , 'directory': "fig_ED5"
          , 'body_color_list': OL_COLOR.PALE_AND_YELLOW.rgba
          , 'body_color_order': [0, 1, 2]
          , 'color_by':'type'
          , 'n_vis': {}
          , 'view': 'Equator_slice'
        }
      , "Fig4c-1_TmY_Group1": {
            'columnar_list': ['TmY3', 'TmY4', 'TmY5a', 'TmY9a']
          , 'list_bids_to_plot': [42010, 91611, 43032, 52346]
          , 'hex_assign': []
          , 'text_placement':[0.78, 0.95, 0,-0.06]
          , 'replace': {}
          , 'directory': "fig_4"
          , 'body_color_list':
                OL_COLOR.OL_TYPES.rgba + OL_COLOR.OL_DARK_TYPES.rgba
              + OL_COLOR.SIX_MORE_COLORS.rgba
          , 'body_color_order': [0, 3, 6, 9, 12, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14]
          , 'color_by':'type'
          , 'n_vis': {}
          , 'view': 'Equator_slice'
        }
      , "Fig4c-2_TmY_Group2": {
            'columnar_list': ['TmY9b', 'TmY10', 'TmY13', 'TmY14']
          , 'list_bids_to_plot':[55648, 47625, 57663, 21807]
          , 'hex_assign': []
          , 'text_placement':[0.78, 0.95, 0,-0.06]
          , 'replace': {}
          , 'directory': "fig_4"
          , 'body_color_list':
                OL_COLOR.OL_TYPES.rgba + OL_COLOR.OL_DARK_TYPES.rgba
              + OL_COLOR.SIX_MORE_COLORS.rgba
          , 'body_color_order': [0, 3, 6, 9, 12, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14]
          , 'color_by':'type'
          , 'n_vis': {}
          , 'view': 'Equator_slice'
        }
      , "Fig1c_Mi1": {
            'columnar_list': ['Mi1']
          , 'list_bids_to_plot':[
                77952, 49351, 35683, 33110, 57398, 35888, 58862, 34189, 36252, 34057
              , 35840, 36954, 36911, 47967, 39727, 41399, 45664, 79752
            ]
          , 'hex_assign':[]
          , 'text_placement':[0.05, 0.95, 0, 0]
          , 'replace':{}
          , 'directory': "fig_1"
          , 'body_color_list': OL_COLOR.OL_IN_SEQ.rgba
          , 'body_color_order': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
          , 'color_by': 'bid'
          , 'n_vis': {}
          , 'view': 'Equator_slice'
        }
      , "Fig1c_TmY5a": {
            'columnar_list': ['TmY5a']
          , 'list_bids_to_plot':[
                65773, 61841, 74821, 55509, 54914, 53500, 59398, 76285, 46155, 76233, 52887, 53380
              , 51229, 53274, 59774, 53751, 60813, 80862, 103422
            ]
          , 'hex_assign':[]
          , 'text_placement':[0.05, 0.95, 0, 0]
          , 'replace':{}
          , 'directory': "fig_1"
          , 'body_color_list': OL_COLOR.OL_CONN_SEQ.rgba
          , 'body_color_order': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
          , 'color_by': 'bid'
          , 'n_vis': {}
          , 'view': 'Equator_slice'
        }
      , "Fig1c_LC17": {
            'columnar_list': ['LC17']
          , 'list_bids_to_plot':[
                25742, 24409, 28207, 46140, 32917, 29219, 34645, 54277, 40641, 52895, 28605
            ]
          , 'hex_assign':[]
          , 'text_placement':[0.05, 0.95, 0, 0]
          , 'replace':{}
          , 'directory': "fig_1"
          , 'body_color_list': OL_COLOR.OL_VPN_SEQ.rgba
          , 'body_color_order': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
          , 'color_by': 'bid'
          , 'n_vis': {}
          , 'view': 'Equator_slice'
        }
      , "Fig1c_LoVC16": {
            'columnar_list': ['LoVC16']
          , 'list_bids_to_plot':[10029, 10053]
          , 'hex_assign':[]
          , 'text_placement':[0.05, 0.95, 0, 0]
          , 'replace':{}
          , 'directory': "fig_1"
          , 'body_color_list': OL_COLOR.OL_VCN_SEQ.rgba
          , 'body_color_order': [1, 2]
          , 'color_by': 'bid'
          , 'n_vis': {}
          , 'view': 'Equator_slice'
        }
      , "Fig4c-3_TmY_Group3": {
            'columnar_list': ['TmY15', 'TmY16', 'TmY17', 'TmY18']
          , 'list_bids_to_plot':[26151, 23353, 39768, 92012]
          , 'hex_assign': []
          , 'text_placement':[0.78, 0.95, 0,-0.06]
          , 'replace': {}
          , 'directory': "fig_4"
          , 'body_color_list':
                OL_COLOR.OL_TYPES.rgba + OL_COLOR.OL_DARK_TYPES.rgba
              + OL_COLOR.SIX_MORE_COLORS.rgba
          , 'body_color_order': [0, 3, 6, 9, 12, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14]
          , 'color_by': 'type'
          , 'n_vis': {}
          , 'view': 'Equator_slice'
        }
      , "Fig4c-4_TmY_Group4": {
            'columnar_list': ['TmY19a', 'TmY19b', 'TmY20', 'TmY21']
          , 'list_bids_to_plot':[17884, 19702, 49775, 34293]
          , 'hex_assign': []
          , 'text_placement':[0.78, 0.95, 0,-0.06]
          , 'replace': {}
          , 'directory': "fig_4"
          , 'body_color_list':
                OL_COLOR.OL_TYPES.rgba + OL_COLOR.OL_DARK_TYPES.rgba
              + OL_COLOR.SIX_MORE_COLORS.rgba
          , 'body_color_order': [0, 3, 6, 9, 12, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14]
          , 'color_by': 'type'
          , 'n_vis': {}
          , 'view': 'Equator_slice'
        }
      , "Fig2e_Pm2": {
            'columnar_list': ['Pm2a', 'Pm2b']
          , 'list_bids_to_plot': [[21674, 21511], [21497, 23746]]
          , 'hex_assign': []
          , 'text_placement': [0.10, 0.45,0.20, -0.21]
          , 'replace': {}
          , 'directory': "fig_2"
          , 'body_color_list': OL_COLOR.OL_TYPES.rgba
          , 'body_color_order': [1, 2]
          , 'color_by': 'type'
          , 'n_vis': {'npil':'ME', 'lay':9, 'flat': True}
          , 'view': 'medulla_face_on'
        }
      , "Fig5c_Dm3a_h1": {
            'columnar_list': ['Dm3a', 'Dm3a_2']
          , 'list_bids_to_plot':[
                [70592, 76518, 93192, 103934, 104618, 107349, 107430, 108394 , 108597, 109558, 109877, 110224, 110616, 111022, 111093, 111789 , 112311, 112742, 114002, 114083, 114124, 114415, 114913, 115833 , 116286, 116505, 116938, 117281, 117614, 117925, 118126, 118167 , 118224, 118442, 118525, 118574, 119206, 119618, 120078, 120281 , 120390, 121090, 121167, 121193, 121235, 121498, 121783, 121827 , 122096, 122128, 122140, 122174, 122316, 123239, 123275, 123538 , 123839, 123843, 124012, 124406, 124474, 124543, 124740, 125077 , 125156, 125611, 125612, 125822, 126065, 126069, 126070, 126185 , 126272, 126325, 126840, 127410, 127718, 127744, 128003, 128092 , 128298, 128582, 128816, 128893, 129183, 129405, 129408, 129422 , 129650, 130105, 130108, 130385, 131043, 131085, 131209, 131812 , 131816, 131909, 131931, 132026, 132058, 132237, 132377, 133095 , 133346, 133391, 133819, 133845, 133876, 133904, 134003, 134045 , 134508, 134527, 135396, 135438, 135463, 135543, 135867, 136063 , 136238, 136517, 136555, 136683, 136894, 136923, 137028, 137478 , 137757, 137805, 137986, 138067, 138155, 138270, 138294, 138341 , 138417, 138442, 138449, 138651, 138973, 139058, 139916, 139986 , 140171, 140202, 140205, 140235, 140354, 140510, 140600, 140653 , 140662, 140679, 140702, 140722, 140814, 141168, 141742, 141955 , 142111, 142141, 142167, 142216, 142231, 142237, 142286, 142301 , 142387, 142442, 142879, 142886, 143113, 143972, 144086, 144534 , 144551, 144806, 144944, 145029, 145118, 145378, 145828, 146199 , 146248, 146249, 146421, 146460, 146530, 146596, 146655, 146746 , 146781, 147189, 147289, 147488, 147563, 147577, 147828, 148142 , 148157, 148224, 148226, 148365, 148479, 148626, 148644, 148675 , 148957, 148970, 149055, 149175, 149233, 149277, 149304, 149480 , 149605, 149967, 150129, 150556, 150693, 150838, 150945, 150959 , 151038, 151221, 151305, 151586, 151759, 152107, 152377, 152547 , 152632, 152727, 152765, 152956, 153274, 153308, 153766, 153802 , 153890, 154130, 154146, 154278, 154362, 154460, 154498, 154635 , 154769, 155023, 155057, 155252, 156456, 156612, 156741, 156852 , 157161, 157311, 157427, 157473, 157497, 157535, 157577, 157799 , 157856, 158206, 158237, 158484, 158996, 159594, 159868, 159967 , 160473, 161150, 161620, 162431, 163104, 164540, 165730, 166362 , 166382, 166735, 167159, 167807, 168376, 170240, 170958, 171563 , 172764, 175217, 175903, 178538, 179444, 179483, 180819, 183312 , 183723, 194672, 213540, 223628, 526113]
              , [78191, 91187, 98262, 100001, 109069, 110102, 110902, 111128 , 111951, 113485, 113622, 115639, 116240, 116310, 116822, 116969 , 117411, 117479, 117608, 117625, 117821, 117974, 118179, 118229 , 118786, 118897, 119045, 119255, 119418, 119600, 120018, 120144 , 120330, 120409, 120776, 120845, 121028, 121522, 121615, 121626 , 121877, 121907, 122034, 122559, 122579, 123327, 123484, 123882 , 124035, 124050, 124094, 124200, 124226, 124539, 124654, 124850 , 125122, 125183, 125226, 125250, 125416, 125539, 125619, 125917 , 125920, 126137, 126293, 126303, 126373, 126459, 126580, 127140 , 127385, 127553, 127587, 127649, 127664, 127698, 127746, 127993 , 128001, 128302, 128333, 128665, 129057, 129131, 129254, 129539 , 129673, 130272, 130313, 130548, 130625, 130943, 131486, 131697 , 131841, 131857, 132028, 132877, 133426, 133659, 133850, 133858 , 133929, 133951, 133954, 134192, 134268, 134572, 134582, 135124 , 135128, 135229, 135392, 135700, 135737, 135848, 136010, 136181 , 136402, 136660, 136749, 136764, 137380, 137565, 137623, 137628 , 137732, 137799, 137978, 138386, 138577, 139161, 139162, 139172 , 139177, 139490, 139533, 139577, 139580, 139912, 140338, 140343 , 140563, 140568, 140686, 140700, 140720, 140840, 140917, 141077 , 141236, 141658, 141743, 141775, 141843, 142081, 142143, 142279 , 142363, 142366, 142794, 142916, 142957, 142998, 143110, 143138 , 143159, 143188, 143944, 143960, 143962, 143964, 143993, 144150 , 144176, 144218, 144567, 144708, 144923, 144958, 145045, 145840 , 145861, 145875, 145943, 146011, 146133, 146367, 146703, 146796 , 146798, 147278, 147369, 147614, 147855, 147910, 147928, 148187 , 148219, 148527, 148637, 148791, 149130, 149143, 149206, 149246 , 149495, 149527, 149768, 149852, 149938, 150269, 150399, 150425 , 150432, 150500, 150604, 150902, 151004, 151060, 151434, 151453 , 152235, 152492, 152508, 152751, 153759, 154154, 154361, 154630 , 155041, 155092, 155202, 155381, 155484, 155503, 155557, 155995 , 156124, 156604, 156781, 157100, 157117, 157506, 157714, 158257 , 158476, 159047, 159408, 159490, 159540, 159593, 159672, 160227 , 160983, 161481, 161666, 161758, 161866, 162111, 162433, 163866 , 163921, 164361, 164622, 164971, 165102, 165157, 165161, 165453 , 165953, 166421, 168724, 169448, 169692, 172863, 173713, 175739 , 177013, 177389, 178635, 180932, 181500, 181583, 181902, 184357 , 186410, 189468, 190436, 191261, 192349, 192781, 193181, 195996 , 196119, 203064, 544164, 545573]
            ]
          , 'hex_assign':[]
          , 'text_placement':[0.05, 0.95, 0, 0]
          , 'replace':{}
          , 'directory': "fig_5"
          , 'body_color_list':
                OL_COLOR.OL_IN_SEQ.rgba
          , 'body_color_order': [2, 0]
          , 'color_by': 'type'
          , 'n_vis': {'npil':'ME','lay':3, 'flat': True}
          , 'view': 'medulla_face_on'
        }
      , "Fig5c_Dm3b_h2": {
            'columnar_list': ['Dm3b1', 'Dm3b']
          , 'list_bids_to_plot':[
                  [100210, 101826, 103635, 103994, 104168, 104239, 104464, 105008, 105354, 109364, 109591, 109923, 110275, 110285, 110307, 110461, 111592, 112719, 113581, 113632, 114071, 114282, 115035, 115762, 115791, 116119, 116282, 116331, 116390, 116669, 116753, 116797, 117263, 117669, 117872, 118273, 118308, 118660, 119380, 119707, 119940, 120317, 121198, 121208, 121281, 121370, 121464, 121534, 121684, 121856, 122065, 122087, 122196, 122227, 122520, 124566, 124729, 124804, 124942, 124991, 125446, 125787, 125929, 126004, 126123, 126304, 126608, 126777, 127157, 127245, 127297, 127911, 127971, 128023, 128155, 128425, 128445, 128464, 128555, 128577, 128617, 128792, 128944, 129350, 129450, 129473, 129856, 129916, 130045, 130050, 130282, 130325, 130540, 130772, 130860, 131075, 131248, 131331, 131371, 131571, 131898, 132078, 132091, 132259, 132282, 132562, 132706, 132743, 133119, 133234, 133263, 133314, 133482, 133486, 133559, 133631, 133811, 134101, 134114, 134140, 134150, 134167, 134337, 134356, 134637, 134697, 135074, 135156, 135425, 135428, 135647, 135829, 135907, 135919, 136170, 136332, 136679, 136784, 136911, 137245, 137461, 137558, 137677, 137900, 138074, 139015, 139189, 139255, 139304, 139316, 139442, 139651, 139730, 139860, 140523, 140678, 140997, 141367, 141434, 141553, 141945, 141954, 142444, 142611, 142800, 143096, 143193, 143232, 143254, 143400, 143511, 143530, 143630, 143654, 143660, 143783, 143839, 143916, 144058, 144130, 144204, 144229, 144323, 144406, 144429, 144875, 144911, 144992, 145303, 145325, 145402, 145410, 145682, 145740, 145836, 146055, 146183, 146296, 146563, 146615, 146924, 147311, 147961, 148193, 148335, 148657, 148878, 148982, 149042, 149188, 149218, 149794, 149955, 150067, 150120, 150527, 150529, 150565, 150786, 150861, 151168, 151338, 151675, 151989, 152348, 152402, 152452, 152526, 152715, 152944, 153228, 153262, 153332, 153561, 153690, 154117, 154291, 154892, 155428, 155516, 155715, 156341, 157759, 158719, 159597, 160032, 160268, 160391, 160782, 161171, 161688, 161971, 162660, 163842, 168198, 170896, 172426, 172941, 173446, 173691, 173866, 174080, 174268, 174848, 177779, 184200, 184234, 184673, 187617, 193787, 194592, 197023, 198534, 206717, 206931, 207418, 210198, 212005, 214624, 526242, 549334, 554901, 554978]
                , [45426, 55890, 58282, 98940, 98948, 101659, 102794, 103341, 103375, 103627, 106371, 106450, 106662, 107886, 107970, 108909, 108911, 109224, 109461, 110229, 110975, 111097, 111977, 112408, 112551, 113095, 114023, 114360, 114780, 114859, 115352, 115688, 115979, 116020, 116337, 116425, 116697, 116922, 117233, 117790, 117839, 118263, 118584, 119028, 119076, 119310, 119584, 119671, 120000, 120413, 120599, 120822, 120913, 121030, 121333, 121417, 121537, 121610, 121699, 121818, 121860, 122142, 122462, 122618, 122669, 123164, 123567, 123892, 124016, 124635, 124663, 124949, 124984, 125335, 125357, 125560, 125573, 125827, 125859, 125879, 127034, 127174, 127341, 127386, 127400, 127417, 127419, 127875, 127966, 128018, 128200, 128684, 128956, 128964, 129286, 129523, 129530, 129833, 130487, 130546, 130770, 130862, 131059, 131222, 131294, 131337, 131583, 131614, 131659, 131791, 131920, 131978, 132115, 132187, 132254, 132447, 132693, 132739, 133199, 133303, 133378, 133620, 133736, 133807, 134450, 134512, 134536, 134748, 134797, 135340, 135416, 135485, 135824, 135999, 136042, 136054, 136418, 136621, 136727, 137045, 137145, 137280, 137288, 137311, 137383, 137518, 137807, 137860, 137897, 138151, 138181, 138242, 138383, 138669, 138713, 138880, 138893, 139300, 139320, 139480, 139523, 139724, 139871, 140138, 140176, 140316, 140348, 140428, 140469, 140587, 140725, 140985, 141278, 141374, 141459, 141516, 141637, 141842, 141963, 142020, 142053, 142242, 142263, 142337, 142494, 142541, 142706, 142717, 143002, 143060, 143105, 143124, 143495, 143500, 144022, 144097, 144170, 144287, 144325, 144601, 144665, 145071, 145132, 145137, 145262, 145305, 145328, 145502, 145839, 146013, 146163, 146575, 146708, 146982, 147117, 147191, 147281, 147287, 147303, 147315, 147558, 147726, 147877, 148040, 148223, 148679, 148757, 148848, 149185, 149322, 149652, 149677, 149995, 150247, 150356, 150479, 151081, 151280, 151312, 152110, 152167, 152363, 152466, 152589, 152744, 152924, 152938, 152953, 153503, 154014, 154747, 156410, 156589, 157660, 157880, 159820, 160855, 162154, 163280, 164017, 168309, 169522, 169739, 171878, 172417, 174934, 179377, 184976, 185302, 185748, 190825, 192766, 192774, 192813, 196837, 197160, 197332, 198022, 198837, 201422, 203668, 207850, 209943, 529923, 532162, 546454, 553531, 554482]
            ]
          , 'hex_assign':[]
          , 'text_placement':[0.05, 0.95, 0, 0]
          , 'replace':{}
          , 'directory': "fig_5"
          , 'body_color_list':
                OL_COLOR.OL_IN_SEQ.rgba
          , 'body_color_order': [2, 0]
          , 'color_by': 'type'
          , 'n_vis': {'npil':'ME','lay':3, 'flat': True}
          , 'view': 'medulla_face_on'
        }
      , "Fig5c_Dm3c_h3": {
            'columnar_list': ['Dm3c1', 'Dm3c']
          , 'list_bids_to_plot':[
                  [39682, 73204, 80150, 89283, 96197, 97041, 97995, 99905, 100367, 106773, 107263, 107695, 107849, 108621, 109223, 110363, 110489, 111299, 111781, 111881, 112491, 112651, 112726, 114250, 114400, 115343, 115661, 117054, 119158, 119231, 120821, 120874, 121298, 121340, 122252, 122800, 123206, 123403, 124141, 124271, 124525, 125935, 126509, 127022, 127658, 128610, 128733, 128836, 128874, 129854, 130127, 130284, 130840, 131788, 131851, 132715, 132751, 133162, 133813, 134046, 134194, 134362, 134599, 134785, 135094, 135136, 135321, 135497, 135505, 135733, 135998, 136097, 136288, 136615, 136691, 136996, 137059, 137611, 137888, 138566, 138570, 138666, 138697, 138758, 138804, 138946, 139091, 139405, 140349, 140391, 140423, 140494, 140570, 141040, 141125, 141272, 141589, 141738, 141946, 141964, 142249, 142377, 142695, 142771, 143090, 143340, 143560, 143756, 143989, 144018, 144392, 144935, 145075, 145205, 145897, 146101, 146107, 146256, 146325, 146761, 146884, 147680, 147813, 147946, 147985, 148103, 148323, 148422, 148426, 148516, 148664, 149009, 149097, 149715, 149843, 149893, 150406, 150809, 151273, 151333, 153844, 154403, 155129, 155145, 155590, 155647, 155704, 155955, 156339, 157104, 157608, 157911, 158146, 158409, 159550, 159619, 160033, 160125, 160431, 160732, 161156, 161894, 163895, 164300, 164918, 164996, 165299, 165343, 165348, 167028, 167503, 168610, 168826, 168998, 169662, 169747, 171162, 171799, 171913, 172182, 174589, 176430, 176895, 178392, 179491, 180375, 180761, 181542, 182206, 186496, 186672, 187519, 187636, 189234, 189839, 190449, 190765, 191170, 191792, 194259, 198976, 203851, 209680, 210586, 216665, 541399]
                , [44401, 47618, 89778, 90339, 90457, 94604, 98018, 98342, 99641, 99713, 103300, 103404, 103498, 105530, 105797, 106081, 107098, 107268, 108316, 109124, 109209, 109594, 109913, 110313, 110558, 110957, 111528, 111718, 111752, 111886, 114622, 115360, 115531, 115701, 115845, 116519, 116557, 117266, 117573, 119020, 119080, 119195, 119443, 119861, 120012, 120684, 120713, 120870, 121556, 121569, 122335, 122839, 123483, 123676, 124773, 125159, 125522, 125561, 125842, 125979, 126856, 126988, 127228, 127926, 128303, 128420, 128774, 129719, 129806, 131693, 131824, 131998, 132221, 133241, 133648, 133869, 133915, 133998, 134597, 134740, 134825, 134901, 135191, 135630, 136002, 136882, 137683, 137910, 137911, 138054, 138170, 138185, 138769, 139446, 140140, 141208, 141829, 141941, 141953, 142248, 142397, 142744, 142780, 142976, 142978, 143076, 143132, 143293, 143411, 143474, 143562, 143638, 144010, 144384, 145185, 145399, 145569, 145661, 146489, 146510, 146649, 146688, 146730, 146732, 146800, 147054, 147057, 147321, 148018, 148024, 148195, 148210, 148257, 148283, 149094, 149472, 149683, 149858, 150059, 150206, 151055, 151160, 151307, 151557, 151679, 152067, 152187, 152570, 152666, 152701, 152863, 153157, 153530, 154605, 154696, 154766, 156027, 156937, 156966, 157388, 159212, 159243, 159334, 159351, 159622, 159710, 159906, 161511, 161683, 162717, 163759, 164258, 164345, 164418, 164509, 164968, 166151, 168810, 171272, 173399, 175120, 176641, 177925, 180170, 183381, 183578, 184177, 185156, 191624, 193431, 198974, 204605, 211616, 549909, 557230]
            ]
          , 'hex_assign':[]
          , 'text_placement':[0.05, 0.95, 0, 0]
          , 'replace':{}
          , 'directory': "fig_5"
          , 'body_color_list':
                OL_COLOR.OL_IN_SEQ.rgba
          , 'body_color_order': [2, 0]
          , 'color_by': 'type'
          , 'n_vis': {'npil':'ME','lay':3, 'flat': True}
          , 'view': 'medulla_face_on'
        }
      , "Fig5f_MeVP10": {
            'columnar_list': ['MeVP10']
          , 'list_bids_to_plot':[
                32594, 32929, 33757, 34435, 34502
              , 34639, 35766, 37671, 41321
              , 41409, 41848, 42525, 42911, 46146
              , 46537, 47163, 49326, 50050, 51239, 52515
              , 52891, 52986, 53284, 53311, 54358, 55952, 58332
              , 60113, 64162, 64658, 66093, 67628, 68046
            ]
          , 'hex_assign':[]
          , 'text_placement':[0.05, 0.95, 0, 0]
          , 'replace':{}
          , 'directory': "fig_5"
          , 'body_color_list':
                OL_COLOR.OL_TYPES.rgba + OL_COLOR.OL_DARK_TYPES.rgba
              + OL_COLOR.SIX_MORE_COLORS.rgba
          , 'body_color_order': [0, 3, 6, 9, 12, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14]
          , 'color_by': 'bid'
          , 'n_vis': {'npil':'ME', 'lay':6, 'flat': True}
          , 'view': 'medulla_face_on'
        }
      , "Fig5e_Dm4": {
            'columnar_list': ['Dm4']
          , 'list_bids_to_plot':[
                16333,  16413,  16678,  16726,  16991,  17145,  17235,  17345
              , 17636,  17806,  17913,  17999,  18006,  18412,  18467,  18530
              , 18541,  18833,  19111,  19131,  19370,  19383,  19430,  19435
              , 19568,  19734,  19782,  19795,  20035,  20198,  20220,  20439
              , 20519,  20660,  20732,  20813,  21139,  21424,  21585,  21765
              , 21985,  22178,  22363,  22680,  23036,  27208,  40923, 203515
            ]
          , 'hex_assign':[]
          , 'text_placement':[0.05, 0.95, 0, 0]
          , 'replace':{}
          , 'directory': "fig_5"
          , 'body_color_list':
                OL_COLOR.OL_TYPES.rgba + OL_COLOR.OL_DARK_TYPES.rgba
              + OL_COLOR.SIX_MORE_COLORS.rgba
          , 'body_color_order': [0, 3, 6, 9, 12, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14]
          , 'color_by': 'bid'
          , 'n_vis': {'npil':'ME','lay':3, 'flat': True}
          , 'view': 'medulla_face_on'
        }
      , "Fig5e_Dm20": {
            'columnar_list': ['Dm20']
          , 'list_bids_to_plot':[
                18858, 19123, 19401, 19528, 19797, 19876, 20027, 20114, 20169
              , 20342, 20483, 20528, 20701, 20857, 20948, 20952, 21030, 21202
              , 21265, 21493, 21620, 22016, 22042, 22044, 22262, 22334, 22411
              , 22483, 22488, 22500, 22687, 22786, 22865, 23044, 23131, 23534
              , 23579, 23767, 24230, 24383, 25603, 25653, 26982, 28124, 28323
              , 28970, 30713, 33246
            ]
          , 'hex_assign':[]
          , 'text_placement':[0.05, 0.95, 0, 0]
          , 'replace':{}
          , 'directory': "fig_5"
          , 'body_color_list':
                OL_COLOR.OL_TYPES.rgba + OL_COLOR.OL_DARK_TYPES.rgba
              + OL_COLOR.SIX_MORE_COLORS.rgba
          , 'body_color_order': [0, 3, 6, 9, 12, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14]
          , 'color_by': 'bid'
          , 'n_vis':{'npil':'ME','lay':3, 'flat': True}
          , 'view': 'medulla_face_on'
        }
      , "Fig5f_l-LNv": {
           'columnar_list': ['l-LNv']
          , 'list_bids_to_plot':[10870, 11114, 11212, 11715]
          , 'hex_assign':[]
          , 'text_placement':[0.05, 0.95, 0, 0]
          , 'replace':{}
          , 'directory': "fig_5"
          , 'body_color_list':
                OL_COLOR.OL_TYPES.rgba + OL_COLOR.OL_DARK_TYPES.rgba
              + OL_COLOR.SIX_MORE_COLORS.rgba
          , 'body_color_order': [0, 3, 6, 9, 12, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14]
          , 'color_by': 'bid'
          , 'n_vis': {'npil':'ME', 'lay':1, 'flat': True}
          , 'view': 'medulla_face_on'
        }
      , "Fig5a-3_Pathway_1": {
            'columnar_list': ['Tm1', 'Tm2', 'TmY4', 'TmY9a', 'TmY9b','LC11','LC15']
          , 'list_bids_to_plot':[48193, 45468, 56703, 62719, 62635, 24953, 31391]
          , 'hex_assign':[]
          , 'text_placement':[0.78, 0.95, 0,-0.06]
          , 'replace':{}
          , 'directory': "fig_5"
          , 'body_color_list':
                OL_COLOR.PALE_AND_YELLOW.rgba + OL_COLOR.OL_CONN_SEQ.rgba
              + OL_COLOR.OL_IN_SEQ.rgba + OL_COLOR.OL_VPN_SEQ.rgba
          , 'body_color_order': [3, 6, 4, 0, 5, 13, 14]
          , 'color_by': 'type'
          , 'n_vis': {}
          , 'view': 'Equator_slice'
        }
      , "Fig5_Pathway_2": {
            'columnar_list':['L2', 'L3', 'Dm3a', 'Dm3b', 'Dm3c', 'Li16']
          , 'list_bids_to_plot':[33893, 94403, 114083, 116753, 136002, 11394]
          , 'hex_assign':[]
          , 'text_placement':[0.78, 0.95, 0,-0.06]
          , 'replace':{}
          , 'directory': "fig_5"
          , 'body_color_list':
                OL_COLOR.MY_COLOR.rgba + OL_COLOR.OL_CONN_SEQ.rgba
              + OL_COLOR.OL_IN_SEQ.rgba + OL_COLOR.OL_VPN_SEQ.rgba
          , 'body_color_order': [0, 3, 10, 11, 12, 9]
          , 'color_by': 'type'
          , 'n_vis': {}
          , 'view': 'Equator_slice'
        }
    }
    return plots


def get_rend_params(
    camera_or_slice:str
  , view_type:str
) -> dict:
    """
    Helper function to return variables from a spreadsheet in `params/`

    Parameters
    ----------
    camera_or_slice : {'camera', 'slice', 'scalebar'}
        get parameters either for the camera, the slicer, or scalebar
    view_type : str
        must be one of the views defined in the Rendering_parameters.xlsx
        (e.g. Dorsal_view)

    Returns
    -------
    param_dict : dict
        simple dictionary with locations, rotations, and other parameters
    """

    assert camera_or_slice in ['camera', 'slice', 'scalebar'], \
        f"Only 'camera', 'slice', or 'scalebar' are allowed, not {camera_or_slice}"
    rendering_parameters = pd.read_excel(
        Path(find_dotenv()).parent / 'params' / 'Rendering_parameters.xlsx'
    )
    rendering_parameters = rendering_parameters.set_index('type_of_view')

    match camera_or_slice:
        case 'slice':
            param_dict = {
                'loc': {
                    'x': rendering_parameters.loc[view_type,'slice_loc_x']
                  , 'y': rendering_parameters.loc[view_type,'slice_loc_y']
                  , 'z': rendering_parameters.loc[view_type,'slice_loc_z']
                }
              , 'rot': {
                    'x': rendering_parameters.loc[view_type,'slice_rot_x']
                  , 'y': rendering_parameters.loc[view_type,'slice_rot_y']
                  , 'z': rendering_parameters.loc[view_type,'slice_rot_z']
                }
            }
        case 'camera':
            param_dict = {
                'loc': {
                    'x': rendering_parameters.loc[view_type,'cam_loc_x']
                  , 'y': rendering_parameters.loc[view_type,'cam_loc_y']
                  , 'z': rendering_parameters.loc[view_type,'cam_loc_z']
                }
              , 'rot': {
                    'x': rendering_parameters.loc[view_type,'cam_rot_x']
                  , 'y': rendering_parameters.loc[view_type,'cam_rot_y']
                  , 'z': rendering_parameters.loc[view_type,'cam_rot_z']
                }
              , 'res': {
                    'x':rendering_parameters.loc[view_type,'cam_res_x']
                  , 'y':rendering_parameters.loc[view_type,'cam_res_y']
                }
              , 'ortho': rendering_parameters.loc[view_type,'cam_ortho']
            }
        case 'scalebar':
            param_dict = {
              'loc': {
                  'x': rendering_parameters.loc[view_type,'scalebar_loc_x']
                , 'y': rendering_parameters.loc[view_type,'scalebar_loc_y']
                , 'z': rendering_parameters.loc[view_type,'scalebar_loc_z']
              }
              ,'txt_loc': {
                  'x': rendering_parameters.loc[view_type,'sb_text_loc_x']
                , 'y': rendering_parameters.loc[view_type,'sb_text_loc_y']
                , 'z': rendering_parameters.loc[view_type,'sb_text_loc_z']
              }
            }
    return param_dict
