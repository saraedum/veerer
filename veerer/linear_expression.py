r"""
Linear expression and linear constraints

This module provides a common interface to ppl, PyNormaliz and sage to build
polyhedra.
"""
from sage.categories.modules import Modules
from sage.categories.number_fields import NumberFields

from sage.structure.element import ModuleElement, Vector, parent
from sage.structure.parent import Parent
from sage.structure.richcmp import op_LE, op_LT, op_EQ, op_NE, op_GT, op_GE

from sage.rings.integer_ring import ZZ
from sage.rings.rational_field import QQ
from sage.rings.qqbar import AA, number_field_elements_from_algebraics
from sage.rings.real_arb import RealBallField
from sage.rings.real_double import RDF

from sage.arith.functions import lcm
from sage.arith.functions import LCM_list

RBF = RealBallField(64)


_NumberFields = NumberFields()


class LinearExpression(ModuleElement):
    r"""
    EXAMPLES::

        sage: from veerer.linear_expression import LinearExpressions
        sage: L = LinearExpressions(QQ)
        sage: L()
        0
        sage: L({0: 5})
        5*x0
        sage: L({0: 5, 1: -2})
        5*x0 - 2*x1
    """
    def __init__(self, parent, f=None, inhomogeneous_term=None):
        ModuleElement.__init__(self, parent)
        self._f = {} if not f else f
        base_ring = parent.base_ring()
        self._inhomogeneous_term = base_ring.zero() if not inhomogeneous_term else inhomogeneous_term

    def change_ring(self, base_ring):
        r"""
        Return the same linear expression on a different base ring.

        EXAMPLES::

            sage: from veerer.linear_expression import LinearExpressions
            sage: L = LinearExpressions(QQ)
            sage: L({0: 5}, 3)
            5*x0 + 3
        """
        if base_ring is self.base_ring():
            return self
        parent = LinearExpressions(base_ring)
        f = {i: base_ring(coeff) for i, coeff in self._f.items()}
        inhomogeneous_term = base_ring(self._inhomogeneous_term)
        return parent.element_class(parent, f, inhomogeneous_term)

    def is_homogeneous(self):
        return self._inhomogeneous_term.is_zero()

    def _repr_(self):
        if not self._f:
            return str(self._inhomogeneous_term)

        def term_string(i, s):
            if s == '1':
                return 'x%d' % i
            else:
                return '%s*x%d' % (s, i)

        # TODO:nicer string representation with + and -
        ind_coeffs = [(i, str(self._f[i])) for i in sorted(self._f)]
        lin_part = term_string(*ind_coeffs[0])
        for i, s in ind_coeffs[1:]:
            if s[0] == '-':
                lin_part += ' - '
                s = s[1:]
            else:
                lin_part += ' + '
            lin_part += term_string(i, s)
        if self._inhomogeneous_term:
            s = str(self._inhomogeneous_term)
            if s[0] == '-':
                return lin_part + ' - ' + s[1:]
            else:
                return lin_part + ' + ' + s
        else:
            return lin_part

    def _lmul_(self, other):
        f = {i: other * coeff for i, coeff in self._f.items()}
        inhomogeneous_term = other * self._inhomogeneous_term
        return self.parent().element_class(self.parent(), f, inhomogeneous_term)

    def _rmul_(self, other):
        f = {i: coeff * other for i, coeff in self._f.items()}
        inhomogeneous_term = self._inhomogeneous_term * other
        return self.parent().element_class(self.parent(), f, inhomogeneous_term)

    def _add_(self, other):
        data = self._f.copy()
        for i, coeff in other._f.items():
            if i in data:
                data[i] += coeff
                if not data[i]:
                    del data[i]
            else:
                data[i] = coeff
        inhomogeneous_term = self._inhomogeneous_term + other._inhomogeneous_term
        return self.parent().element_class(self.parent(), data, inhomogeneous_term)

    def _sub_(self, other):
        data = self._f.copy()
        for i, coeff in other._f.items():
            if i in data:
                data[i] -= coeff
                if not data[i]:
                    del data[i]
                else:
                    data[i] = -coeff
            else:
                data[i] = -coeff
        inhomogeneous_term = self._inhomogeneous_term - other._inhomogeneous_term
        return self.parent().element_class(self.parent(), data, inhomogeneous_term)

    def __neg__(self):
        return LinearExpression(self.parent(), {i: -c for (i, c) in self._f.items()}, -self._inhomogeneous_term)

    def __le__(self, other):
        return LinearConstraint(op_LE, self, other)

    def __lt__(self, other):
        return LinearConstraint(op_LT, self, other)

    def __eq__(self, other):
        return LinearConstraint(op_EQ, self, other)

    def __ne__(self, other):
        return LinearConstraint(op_NE, self, other)

    def __gt__(self, other):
        return LinearConstraint(op_GT, self, other)

    def __ge__(self, other):
        return LinearConstraint(op_GE, self, other)

    def _richcmp_(self, other, op):
        return LinearExpression(op, self, other)

    def denominator(self):
        r"""
        Return the common denominator of the coefficients of this linear expression.
        """
        return LCM_list([coeff.denominator() for coeff in self._f.values()] + [self._inhomogeneous_term.denominator()])

    def integral(self):
        if self.base_ring() is ZZ:
            return self
        if self.base_ring() is QQ:
            return (self.denominator() * self).change_ring(ZZ)
        raise ValueError('invalid base ring')

    def ppl(self):
        r"""
        EXAMPLES::

            sage: from veerer.linear_expression import LinearExpressions
            sage: L = LinearExpressions(QQ)
            sage: (3 * L.variable(2) - 7/2 * L.variable(5) + 1/3).ppl()
            18*x2-21*x5+2
        """
        from gmpy2 import mpz
        import ppl
        lin = self.integral()
        return sum(mpz(coeff) * ppl.Variable(i) for i, coeff in lin._f.items()) + mpz(lin._inhomogeneous_term)

    def coefficients(self, dim=None, homogeneous=False):
        r"""
        Return the coefficients of this linear form as a list.

        If ``homogeneous`` is ``False`` (default) the first coefficient is the homogeneous term.

        EXAMPLES::

            sage: from veerer.linear_expression import LinearExpressions
            sage: L = LinearExpressions(QQ)
            sage: lin = 3 * L.variable(2) - 7/2 * L.variable(5) + 1/3
            sage: lin.coefficients()
            [-1/3, 0, 0, 3, 0, 0, -7/2]
            sage: lin.coefficients(homogeneous=True)
            [0, 0, 3, 0, 0, -7/2]

            sage: lin.coefficients(10)
            [-1/3, 0, 0, 3, 0, 0, -7/2, 0, 0, 0, 0]
        """
        if dim is None:
            if not self._f:
                dim = 0
            else:
                dim = max(self._f) + 1
        zero = self.base_ring().zero()
        if homogeneous:
            return [self._f.get(i, zero) for i in range(dim)]
        else:
            return [-self._inhomogeneous_term] + [self._f.get(i, zero) for i in range(dim)]


class LinearConstraint:
    r"""
    EXAMPLES::

        sage: from veerer.linear_expression import LinearExpressions
        sage: L = LinearExpressions(QQ)
        sage: 3 * L.variable(0) - 5 * L.variable(2) == 7
        3*x0 - 5*x2 - 7 = 0
    """
    def __init__(self, op, left, right):
        self._expression = left - right
        self._op = op

    @staticmethod
    def _from_ppl(cst):
        f = cst.coefficients()
        inhomogeneous_term = cst.inhomogeneous_term()
        L = LinearExpressions(ZZ)
        lin = L.element_class(L, {i: c for i, c in enumerate(f) if c}, inhomogeneous_term)
        if cst.is_equality():
            return LinearConstraint(op_EQ, lin, ZZ.zero())
        elif cst.is_inequality():
            return LinearConstraint(op_GE, lin, ZZ.zero())
        raise ValueError('invalid constraint cst={}'.format(cst))

    def __repr__(self):
        if self._op == op_LE:
            op = '<='
        elif self._op == op_LT:
            op = '<'
        elif self._op == op_EQ:
            op = '='
        elif self._op == op_NE:
            op = '!='
        elif self._op == op_GT:
            op = '>'
        elif self._op == op_GE:
            op = '>='
        else:
            raise RuntimeError
        return '{} {} 0'.format(self._expression, op)

    def is_homogeneous(self):
        return self._expression.is_homogeneous()

    def coefficients(self, dim=None, homogeneous=False):
        return self._expression.coefficients(dim, homogeneous)

    def ppl(self):
        r"""
        EXAMPLES::

            sage: from veerer.linear_expression import LinearExpressions
            sage: L = LinearExpressions(QQ)
            sage: (3 * L.variable(0) >= 1).ppl()
            3*x0-1>=0
            sage: (3 * L.variable(0) > 1).ppl()
            3*x0-1>0
            sage: (3 * L.variable(0) == 1).ppl()
            3*x0-1==0
            sage: (3 * L.variable(0) < 1).ppl()
            -3*x0+1>0
            sage: (3 * L.variable(0) <= 1).ppl()
            -3*x0+1>=0
        """
        if self._op == op_LE:
            return self._expression.ppl() <= 0
        elif self._op == op_LT:
            return self._expression.ppl() < 0
        elif self._op == op_EQ:
            return self._expression.ppl() == 0
        elif self._op == op_NE:
            return self._expression.ppl() != 0
        elif self._op == op_GT:
            return self._expression.ppl() > 0
        elif self._op == op_GE:
            return self._expression.ppl() >= 0

    def integral(self):
        return LinearConstraint(self._op, self._expression.integral(), ZZ.zero())


class ConstraintSystem:
    r"""
    EXAMPLES::

        sage: from veerer.linear_expression import *
        sage: L = LinearExpressions(QQ)
        sage: cs = ConstraintSystem()
        sage: cs.insert(L.variable(0) >= 2)
        sage: cs.insert(2 * L.variable(1) - L.variable(3) <= 18)
        sage: cs
        {x0 - 2 >= 0, 2*x1 - x3 - 18 <= 0}
    """
    def __init__(self):
        self._data = []
        self._dim = None

    def __repr__(self):
        return '{' + ', '.join(map(str, self)) + '}'

    def copy(self):
        cs = ConstraintSystem.__new__(ConstraintSystem)
        cs._data = self._data[:]
        cs._dim = self._dim
        return cs

    def insert(self, constraint):
        if not isinstance(constraint, LinearConstraint):
            raise TypeError('invalid type; expected LinearConstraint but got {}'.format(type(constraint).__name__))
        if self._dim is None:
            self._dim = max(constraint._expression._f) + 1
        else:
            self._dim = max(self._dim, max(constraint._expression._f) + 1)
        self._data.append(constraint)

    def __iter__(self):
        return iter(self._data)

    def ppl(self):
        r"""
        EXAMPLES::

            sage: from veerer.linear_expression import LinearExpressions, ConstraintSystem
            sage: L = LinearExpressions(QQ)
            sage: cs = ConstraintSystem()
            sage: cs.insert(2 * L.variable(0) - 3/5 * L.variable(2) >= 1)
            sage: cs.insert(L.variable(0) + L.variable(1) + L.variable(2) == 3)
            sage: cs.ppl()
            Constraint_System {10*x0-3*x2-5>=0, x0+x1+x2-3==0}
        """
        import ppl
        cs = ppl.Constraint_System()
        for constraint in self._data:
            cs.insert(constraint.ppl())
        return cs

    def ieqs_eqns(self, dim=None, homogeneous=False):
        r"""
        EXAMPLES::

            sage: from veerer.linear_expression import LinearExpressions, ConstraintSystem
            sage: L = LinearExpressions(QQ)
            sage: cs = ConstraintSystem()
            sage: cs.insert(2 * L.variable(0) - 3/5 * L.variable(2) >= 1)
            sage: cs.insert(L.variable(3) <= 1)
            sage: cs.insert(L.variable(0) + L.variable(1) + L.variable(2) == 3)
            sage: cs.ieqs_eqns()
            ([[1, 2, 0, -3/5, 0], [-1, 0, 0, 0, -1]], [[3, 1, 1, 1, 0]])
            sage: cs.ieqs_eqns(homogeneous=True)
            ([[2, 0, -3/5, 0], [0, 0, 0, -1]], [[1, 1, 1, 0]])

            sage: ieqs, eqns = cs.ieqs_eqns()
            sage: Polyhedron(ieqs=ieqs, eqns=eqns)
            A 3-dimensional polyhedron in QQ^4 defined as the convex hull of 1 vertex, 2 rays, 1 line
        """
        if dim is None:
            dim = self._dim
        ieqs = []
        eqns = []
        for constraint in self._data:
            if constraint._op == op_GE:
                ieqs.append(constraint._expression.coefficients(dim, homogeneous))
            elif constraint._op == op_LE:
                ieqs.append((-constraint._expression).coefficients(dim, homogeneous))
            elif constraint._op == op_EQ:
                eqns.append(constraint._expression.coefficients(dim, homogeneous))
            else:
                raise ValueError('invalid constraint {}'.format(constraint))
        return ieqs, eqns

    def is_linear_subspace(self):
        return all(constraint._op == op_EQ and constraint._expression._inhomogeneous_term.is_zero() for constraint in self)

    def linear_generators_matrix(self, dim=None):
        from sage.matrix.constructor import matrix
        if dim is None:
            dim = self._dim
        if not self.is_linear_subspace():
            raise ValueError('not a linear subspace')
        mat = matrix([constraint._expression.coefficients(dim)[1:] for constraint in self])
        return mat.right_kernel_matrix()

    def integral(self):
        cs = ConstraintSystem()
        for constraint in self:
            cs.insert(constraint.integral())
        return cs

    def cone(self, backend='sage'):
        r"""
        EXAMPLES::

            sage: from veerer.linear_expression import LinearExpressions, ConstraintSystem

            sage: L = LinearExpressions(QQ)
            sage: cs = ConstraintSystem()
            sage: cs.insert(L.variable(0) >= 0)
            sage: cs.insert(L.variable(1) >= 0)
            sage: cs.insert(L.variable(2) >= 0)
            sage: cs.insert(L.variable(3) >= 0)
            sage: cs.insert(L.variable(0) + 1/5 * L.variable(1) - 2/3 * L.variable(2) + 6 * L.variable(3) == 0)

            sage: cone_sage = cs.cone('sage')
            sage: cone_ppl = cs.cone('ppl')
            sage: cone_nmz = cs.cone('normaliz-QQ')
            sage: cones = [cone_sage, cone_ppl, cone_nmz]
            sage: assert all(cone.space_dimension() == 4 for cone in cones)
            sage: assert all(cone.affine_dimension() == 3 for cone in cones)
            sage: assert all(sorted(cone.ieqs()) == [[-15, -3, 10, 0], [0, 1, 0, 0], [1, 0, 0, 0]] for cone in cones)
            sage: assert all(sorted(cone.eqns()) == [[15, 3, -10, 90]] for cone in cones)
            sage: assert all(sorted(cone.rays()) == [[0, 0, 9, 1], [0, 10, 3, 0], [2, 0, 3, 0]] for cone in cones)

        An example over a number field::

            sage: K = NumberField(x^2 - x - 1, 'phi', embedding=(AA(5).sqrt() + 1)/2)
            sage: phi = K.gen()
            sage: L = LinearExpressions(K)
            sage: cs = ConstraintSystem()
            sage: cs.insert(L.variable(0) - phi * L.variable(1) >= 0)
            sage: cs.insert(L.variable(0) + L.variable(1) == 0)
            sage: cone_sage = cs.cone('sage')
            sage: cone_nmz = cs.cone('normaliz-NF')

        Note that contrarily to sage, normaliz does some simplifications::

            sage: cone_sage.ieqs()
            [[phi + 1, 0]]
            sage: cone_nmz.ieqs()
            [[1, 0]]

            sage: cone_sage.eqns()
            [[1, 1]]
            sage: cone_nmz.eqns()
            [[1, 1]]
        """
        # homogeneous case
        if not all(constraint.is_homogeneous() for constraint in self):
            raise ValueError

        if backend == 'ppl':
            import ppl
            return Cone_ppl(QQ, ppl.C_Polyhedron(self.ppl()))
        elif backend == 'sage':
            from sage.geometry.polyhedron.constructor import Polyhedron
            ieqs, eqns = self.ieqs_eqns()
            P = Polyhedron(ieqs=ieqs, eqns=eqns)
            return Cone_sage(P.base_ring(), P)
        elif backend == 'normaliz-QQ' or 'normaliz-NF':
            nmz_data = {}
            if backend == 'normaliz-QQ':
                base_ring = QQ
                ieqs, eqns = self.integral().ieqs_eqns(homogeneous=True)
            else:
                ieqs, eqns = self.ieqs_eqns(homogeneous=True)
                base_ring = embedded_nf([x for l in (ieqs + eqns) for x in l])
                ieqs = [[str(base_ring(x)) for x in ieq] for ieq in ieqs]
                eqns = [[str(base_ring(x)) for x in eqn] for eqn in eqns]
                nmz_data["number_field"] = nmz_number_field_data(base_ring)
            nmz_data["inequalities"] = ieqs
            nmz_data["equations"] = eqns
            from PyNormaliz import NmzCone
            return Cone_normaliz(base_ring, NmzCone(**nmz_data))
        else:
            raise RuntimeError


class LinearExpressions(Parent):
    r"""
    EXAMPLES::

        sage: from veerer.linear_expression import LinearExpressions
        sage: LinearExpressions(QQ)
        Linear expressions over Rational Field
        sage: LinearExpressions(QuadraticField(5))
        Linear expressions over Number Field in a with defining polynomial x^2 - 5 with a = 2.236067977499790?
    """
    Element = LinearExpression

    def __init__(self, base_ring):
        Parent.__init__(self, base=base_ring, category=Modules(base_ring))
        self._populate_coercion_lists_(coerce_list=[base_ring])

    def _repr_(self):
        return 'Linear expressions over {}'.format(self.base_ring())

    def variable(self, i):
        r"""
        EXAMPLES::

        sage: from veerer.linear_expression import *
        sage: L = LinearExpressions(QQ)
        sage: L.variable(0)
        x0
        sage: L.variable(1)
        x1
        sage: 5 * L.variable(2) - 3 * L.variable(7)
        5*x2 - 3*x7
        """
        return LinearExpression(self, {i: self.base_ring().one()})

    def _element_constructor_(self, *args):
        if not args:
            return LinearExpression(self)
        elif len(args) == 1:
            base_ring = self.base_ring()
            data = args[0]
            if isinstance(data, (tuple, list, Vector)):
                # TODO: should we consider vector as homogeneous or inhomogeneous?
                f = {i: base_ring(coeff) for i, coeff in enumerate(data)}
                inhomogeneous_term = base_ring.zero()
            elif isinstance(data, dict):
                f = {int(i): base_ring(coeff) for i, coeff in data.items()}
                inhomogeneous_term = base_ring.zero()
            elif data in base_ring:
                f = {}
                inhomogeneous_term = base_ring(data)
            else:
                raise ValueError('can not construct linear expression from {}'.format(data))
            return LinearExpression(self, f, inhomogeneous_term)

        elif len(args) == 2:
            data0 = args[0]
            data1 = args[1]
            base_ring = self.base_ring()

            if isinstance(data0, (tuple, list, Vector)):
                data0 = {i: base_ring(coeff) for i, coeff in enumerate(data)}
            elif isinstance(data0, dict):
                data0 = {int(i): base_ring(coeff) for i, coeff in data0.items()}
            else:
                raise ValueError('invalid first argument {}'.format(data0))
            data1 = base_ring(data1)
            return LinearExpression(self, data0, data1)
        else:
            raise ValueError('can not construct linear expression from {}'.format(args))



class Cone:
    _name = 'none'

    def __init__(self, base_ring, cone):
        self._base_ring = base_ring
        self._cone = cone

    def __repr__(self):
        return 'Cone of dimension {} in ambient dimension {} made of {} facets (backend={})'.format(
                self.affine_dimension(), self.space_dimension(), len(self.ieqs()), self._name)

    def space_dimension(self):
        raise NotImplementedError

    def affine_dimension(self):
        raise NotImplementedError

    def ieqs(self):
        raise NotImplementedError

    def eqns(self):
        raise NotImplementedError

    def rays(self):
        raise NotImplementedError

    def lines(self):
        raise NotImplementedError

    def facets(self):
        raise NotImplementedError

    def add_constraints(self, cs):
        raise NotImplementedError

    def satisfies(self, constraint):
        if constraint.is_equality():
            if not constraint.is_homogeneous():
                return False
            eqns = self.eqns()
            return matrix(eqns + constraint.coefficients(homogeneous=True)).rank() == len(eqns)

class Cone_ppl(Cone):
    r"""
    PPL cone wrapper.
    """
    _name = 'ppl'

    def __hash__(self):
        ieqs, eqns = self.ieqs_eqns()
        ieqs.sort()
        eqns.sort()
        return hash(tuple(tuple(ieq) for ieq in ieqs), tuple(tuple(eqn) for eqn in eqns))

    def space_dimension(self):
        return self._cone.space_dimension()

    def affine_dimension(self):
        return self._cone.affine_dimension()

    def ieqs(self):
        return [[ZZ(x) for x in c.coefficients()] for c in self._cone.minimized_constraints() if c.is_inequality()]

    def eqns(self):
        return [[ZZ(x) for x in c.coefficients()] for c in self._cone.minimized_constraints() if c.is_equality()]

    def rays(self):
        return [[ZZ(x) for x in g.coefficients()] for g in self._cone.minimized_generators() if g.is_ray()]

    def lines(self):
        return [[ZZ(x) for x in g.coefficients()] for g in self._cone.minimized_generators() if g.is_line()]

    def add_constraints(self, cs):
        import ppl
        cone = ppl.C_Polyhedron(P)
        if isinstance(cs, LinearConstraint):
            cone.add_constraint(cs.ppl())
        elif isinstance(cs, ConstraintSystem):
            for constraint in cs:
                cone.add_constraint(constraint.ppl())
        else:
            raise TypeError
        return Cone_ppl(cone)


class Cone_sage(Cone):
    r"""
    Sage cone wrapper.
    """
    _name = 'sage'

    def __hash__(self):
        return hash(self._cone)

    def space_dimension(self):
        return self._cone.ambient_dimension()

    def affine_dimension(self):
        return self._cone.dimension()

    def ieqs(self):
        return [ieq[1:] for ieq in self._cone.inequalities_list()]

    def eqns(self):
        return [eqn[1:] for eqn in self._cone.equations_list()]

    def rays(self):
        return self._cone.rays_list()

    def lines(self):
        return self._cone.lines_list()

    def add_constraints(self, cs):
        from sage.geometry.polyhedron.constructor import Polyhedron
        if isinstance(cs, LinearConstraint):
            constraint = cs
            cs = ConstraintSystem()
            cs.insert(constraint)
        elif isinstance(cs, ConstraintSystem):
            pass
        else:
            raise TypeError
        ieqs, eqns = cs.ieqs_eqns(self._cone.ambient_dimension())
        new_cone = self._cone.intersection(Polyhedron(ieqs=ieqs, eqns=eqns))
        return Cone_sage(new_cone)

class NFElementHandler:
    def __init__(self, nf):
        self._nf = nf

    def __call__(self, l):
        nf = self._nf
        l = list(l) + [0] * (nf.degree() - len(l))
        l = nf(l)
        return l

class Cone_normaliz(Cone):
    r"""
    Normaliz cone wrapper.
    """
    _name = 'normaliz'

    @property
    def _rational_handler(self):
        return lambda l: QQ((l[0], l[1]))

    @property
    def _nfelem_handler(self):
        return NFElementHandler(self._base_ring)

    def _nmz_result(self, prop):
        from PyNormaliz import NmzResult
        return NmzResult(self._cone, prop,
                         RationalHandler=self._rational_handler,
                         NumberfieldElementHandler=self._nfelem_handler)

    def __hash__(self):
        ieqs, eqns = self.ieqs_eqns()
        ieqs.sort()
        eqns.sort()
        return hash(tuple(tuple(ieq) for ieq in ieqs), tuple(tuple(eqn) for eqn in eqns))

    def space_dimension(self):
        return self._nmz_result("EmbeddingDim")

    def affine_dimension(self):
        return self._nmz_result("Rank")

    def ieqs(self):
        return self._nmz_result("SupportHyperplanes")

    def eqns(self):
        return self._nmz_result("Equations")

    def rays(self):
        return self._nmz_result("ExtremeRays")

    def lines(self):
        raise NotImplementedError


def embedded_nf(l):
    from sage.structure.element import get_coercion_model
    cm = get_coercion_model()
    K = cm.common_parent(*l)
    if K in _NumberFields:
        if not RDF.has_coerce_map_from(K):
            raise ValueError("invalid base ring {} (no embedding)".format(K))
        return K
    elif K == AA:
        K, ll, hom = number_field_elements_from_algebraics(l, embedded=True, minimal=True)
        if K == QQ:
            raise ValueError('rational base ring')
        return K

def nmz_number_field_data(base_ring):
    gen = base_ring.gen()
    s_gen = str(gen)
    emb = RBF(gen)
    return [str(base_ring.polynomial()).replace("x", s_gen), s_gen, str(emb)]