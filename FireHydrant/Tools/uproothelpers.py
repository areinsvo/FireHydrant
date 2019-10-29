#!/usr/bin/env python
"""Helper functions to facilitate uproot/awkward-array operations.
"""
# from awkward import JaggedArray
import awkward


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

    outcontent = content[nnidx.flatten(axis=1)].flatten()
    outnest = awkward.JaggedArray.fromoffsets(nnidx.flatten().offsets, outcontent)
    out = awkward.JaggedArray.fromoffsets(nnidx.offsets, outnest)
    return out