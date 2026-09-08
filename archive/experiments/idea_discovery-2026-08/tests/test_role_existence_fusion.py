import numpy as np

from scripts.idea_discovery.fuse_role_existence import dual_axis_active


def test_generic_component_requires_role_anchor():
    # first column wins on active bins
    role = np.asarray([[.1,.9],[.8,.2],[.1,.9],[.1,.9]])
    generic = np.asarray([[.7,.3],[.7,.3],[.7,.3],[.2,.8]])
    assert dual_axis_active(role, generic, "generic_anchored").tolist() == [1,1,1,0]
