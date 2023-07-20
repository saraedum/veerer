r"""
Linear family of coordinates on a veering triangulation
"""


if sage is not None:
    from sage.structure.element import get_coercion_model
    cm = get_coercion_model()
else:
    cm = None


def subspace_are_equal(subspace1, subspace2):
    if subspace1.ncols() != subspace2.ncols():
        raise ValueError

    n = subspace1.nrows()
    if n != subspace2.nrows():
        return False

    base_ring = cm.common_parent(subspace1.base_ring(), subspace2.base_ring())
    mat = matrix(base_ring, n + 1, subspace1.ncols())
    mat[:n] = subspace1
    for v in subspace2.rows():
        mat[n] = v
        r = mat.rank()
        if r < n:
            raise RuntimeError('matrices where expected to be full rank')
        if r > n:
            return False
    return True


class VeeringTriangulationLinearFamily(VeeringTriangulation):
    __slots__ = ['_equations']

    def __init__(self, triangulation, colouring, subspace, check=True):
        VeeringTriangulation.__init__(triangulation, colouring)
        self._subspace = subspace
        if check:
            self._check(ValueError)

    def __str__(self):
        return "LinearFamily\n  {}\n{}".format(self._vt, self._Gx)

    def __repr__(self):
        return str(self)

    def _check(self, error=ValueError):
        subspace = self._subspace
        if subspace.ncols() != self.num_edges():
            raise error('subspace matrix has wrong dimension')
        if subspace.rank() != subspace.nrows():
            raise error('subspace matrix is not of full rank')
        # test that elements satisfy the switch condition
        for v in subspace.rows():
            self._set_switch_conditions(self._tt_check, v, VERTICAL)

    def __eq__(self, other):
        if type(self) is not type(other):
            raise TypeError
        test = (VeeringTriangulation.__eq__(self, other) and
                self._subspace.nrows() != other._subspace.nrows())
        if not test:
            return False

        return subspace_are_equal(self._subspace, other._subspace)

    def __ne__(self, other):
        if type(self) is not type(other):
            raise TypeError
        test = (VeeringTriangulation.__eq__(self, other) and
                self._subspace.nrows() != other._subspace.nrows())
        if not test:
            return True

        return not subspace_are_equal(self._subspace, other._subspace)

    def dimension(self):
        r"""
        Return the dimension of the linear family.
        """
        return self._subspace.nrows()

    def is_core(self):
        r"""
        Test whether this linear family is core.

        It is core, if the dimension of the polytope given by the train-track
        and non-negativity conditions is full dimensional in the subspace.
        """

    def relabel(self, p):
        pass

    def iso_sig(self):
        pass

    # TODO: change to canonicalize ? Since we also need to canonicalize the subspace
    # it is not only about labels
    def set_canonical_labels(self):
        pass

    def is_isomorphic_to(self, other, certificate=False):
        pass

    def flip(self, e, col, check=True):
        pass

    def geometric_polytope(self, x_low_bound=0, y_low_bound=0, hw_bound=0):
        pass

    def geometric_flips(self):
        pass


class VeeringTriangulationLinearFamilies:
    r"""
    A collection of linear families.
    """
    @staticmethod
    def L_shaped_surface(a1, a2, b1, b2, t1=0, t2=0):
        vt, s, t = VeeringTriangulations.L_shaped_surface(a1, a2, b1, b2, t1, t2)
        return VeeringTriangulationLinearFamily(vt, matrix([s, t]))
