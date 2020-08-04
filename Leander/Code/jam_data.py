import jams
import numpy as np


def get_annotation_metadata():
    # metadata = {
    #     "curator": {
    #       "name": "",
    #       "email": ""
    #     },
    #     "annotator": {},
    #     "version": "",
    #     "corpus": "",
    #     "annotation_tools": "",
    #     "annotation_rules": "",
    #     "validation": "",
    #     "data_source": "MSAF"
    # }
    metadata = jams.AnnotationMetadata(data_source='MSAF')
    return metadata


def get_sandbox():
    # sandbox = {
    #     "boundaries_id": "sf",
    #     "labels_id": null,
    #     "timestamp": "2020/06/06 13:39:32",
    #     "annot_beats": false,
    #     "feature": "cqt",
    #     "framesync": false,
    #     "feature_default": null,
    #     "M_gaussian": 27,
    #     "m_embedded": 3,
    #     "k_nearest": 0.04,
    #     "Mp_adaptive": 28,
    #     "offset_thres": 0.05,
    #     "bound_norm_feats": Infinity,
    #     "hier": false
    # }
    sandbox = jams.Sandbox(
        boundaries_id= "sf",
        labels_id= None,
        timestamp= "2020/06/06 13:39:32",
        annot_beats= False,
        feature= "cqt",
        framesync= False,
        feature_default= None,
        M_gaussian= 27,
        m_embedded= 3,
        k_nearest= 0.04,
        Mp_adaptive= 28,
        offset_thres= 0.05,
        bound_norm_feats= np.Infinity,
        hier= False
    )
    return sandbox


# jam = {
#     "annotations" : {
#         "annotation_metadata" : {
#             "curator" : {
#                 "name" : "",
#                 "email" : ""
#             },
#             "annotator" : {},
#             "version" : "",
#             "corpus" : "",
#             "annotation_tools" : "",
#             "annotation_rules" : "",
#             "validation" : "",
#             "data_source" : "MSAF"
#         },
#         "namespace" : "segment_salami_function",
#         "data" : observations
#         }
#     }