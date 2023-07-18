######################################################################
# This file is part of veerer.
#
#       Copyright (C) 2023 Vincent Delecroix
#
# veerer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# veerer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with veerer. If not, see <https://www.gnu.org/licenses/>.
######################################################################

import pytest

from veerer import VeeringTriangulation, CoreAutomaton, ReducedCoreAutomaton, GeometricAutomaton

@pytest.mark.parametrize("fp, cols",
[("(0,2,~1)(1,~0,~2)", "RBB"),
 ("(0,6,~5)(1,8,~7)(2,7,~6)(3,~1,~8)(4,~2,~3)(5,~0,~4)", "RRRBBBBBB"),
 ("(0,8,~7)(1,11,~10)(2,10,~9)(3,9,~8)(4,~1,~11)(5,~2,~4)(6,~3,~5)(7,~0,~6)", "RRRRBBBBBBBB"),
 ("(0,2,4)(1,~3,~2)(3,~5,~4)(5,~1,~0)", "RBBRBB"),
 ("(0,~8,7)(1,6,~2)(2,~6,~3)(3,5,~4)(4,8,~5)(~7,~1,~0)", "RBBBBRRBB"),
 ("(0,~9,8)(1,10,~2)(2,~6,~3)(3,5,~4)(4,9,~5)(6,11,~7)(7,~10,~8)(~11,~1,~0)", "RBBBBRRBBBRB"),
 ("(0,~11,10)(1,~9,~2)(2,8,~3)(3,9,~4)(4,~8,~5)(5,7,~6)(6,11,~7)(~10,~1,~0)", "RBBBBBBRRRBB"),
 ("(0,8,11)(1,10,~2)(2,~10,~3)(3,~7,~4)(4,6,~5)(5,9,~6)(7,~9,~8)(~11,~1,~0)", "RBBBBBRRBBRB"),
 ("(0,6,8)(1,~11,~2)(2,10,~3)(3,11,~4)(4,~10,~5)(5,~7,~6)(7,~9,~8)(9,~1,~0)", "RBBBBBRBBRRR"),
 ("(0,6,10)(1,9,~2)(2,~9,~3)(3,8,~4)(4,11,~5)(5,~7,~6)(7,~11,~8)(~10,~1,~0)", "RBBBBBRBRRBR")])
def test_automata(fp, cols):
    V = VeeringTriangulation(fp, cols)
    assert V.is_geometric()
    C = CoreAutomaton(V)

    R = ReducedCoreAutomaton(V)
    assert len(R) <= len(C)
    reduced = set()
    for x in C:
        x.forgot_forward_flippable_colour()
        reduced.add(x.iso_sig())
    assert set(x.iso_sig() for x in R) == reduced

    G = GeometricAutomaton(V)
    assert len(G) <= len(C)
    assert set(x.to_string() for x in G) == set(x.to_string() for x in C if x.is_geometric())
