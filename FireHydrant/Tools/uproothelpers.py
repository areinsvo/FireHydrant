#!/usr/bin/env python
"""Helper functions to facilitate uproot/awkward-array operations.
"""
# from awkward import JaggedArray
import awkward
import numpy as np

def NestNestObjArrayToJagged(objarr):
    """uproot read vector<vector<number>> TBranch
       as objectArray, this function convert it
       to JaggedJaggedArray
    """

#     # jaggedArray of lists
#     jaggedList = JaggedArray.fromiter(objarr)
#     # flat to 1 level
#     _jagged = JaggedArray.fromiter(jaggedList.content)

#     return JaggedArray.fromoffsets(jaggedList.offsets, _jagged)
    return awkward.fromiter(objarr)

def fromNestNestIndexArray(content, nnidx):
    """indexing a JaggedArray with a two-level nested index array"""

    if np.any(nnidx.flatten(axis=1).counts):
        try:
            outcontent = content[nnidx.flatten(axis=1)].flatten()
        except:
            print('nnidx:', nnidx)
            print('content:', content)
            raise
        outnest = awkward.JaggedArray.fromoffsets(nnidx.flatten().offsets, outcontent)
        out = awkward.JaggedArray.fromoffsets(nnidx.offsets, outnest)
        return out
    else:
        return nnidx