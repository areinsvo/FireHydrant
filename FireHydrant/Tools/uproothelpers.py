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
