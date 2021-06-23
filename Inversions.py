# type: ignore

import math

import manim as ma
import numpy as np
import numpy.linalg as linalg
from manim.mobject.geometry import ArcPolygon


class ManimUtils:
    @staticmethod
    def create_point(coords: np.ndarray, txt: str, color=ma.WHITE) -> ma.VDict:
        return ma.VDict(
            {
                "dot": ma.Dot(point=coords),
                "txt": ma.MathTex(txt).move_to(coords).shift(ma.UP / 2),
            }
        ).set_color(color)


class MathUtils:
    @staticmethod
    def distance(a: np.ndarray, b: np.ndarray) -> float:
        """Euclidean distance"""
        return linalg.norm(a - b)

    @staticmethod
    def circ_inverse_of(p: np.ndarray, c: np.ndarray, r: float) -> np.ndarray:
        """Calculate the circular inverse of point `p` with respect to
        a circle with center: `c` and radius: `r`.
        The coordinates are given with respect to the default coordinate frame
        with origin: `ma.ORIGIN`.
        `p_norm * prim_norm == r ** 2`
        """

        adj_direction = p - c
        p_norm = linalg.norm(adj_direction)
        prim_norm = (r ** 2) / p_norm

        scale = prim_norm / p_norm
        scale = min(scale, 9223372036854775807)
        return c + scale * adj_direction

    @staticmethod
    def define_circle(p1, p2, p3):
        x1 = p1[0]
        y1 = p1[1]
        x2 = p2[0]
        y2 = p2[1]
        x3 = p3[0]
        y3 = p3[1]
        c = (x1 - x2) ** 2 + (y1 - y2) ** 2
        a = (x2 - x3) ** 2 + (y2 - y3) ** 2
        b = (x3 - x1) ** 2 + (y3 - y1) ** 2
        s = 2 * (a * b + b * c + c * a) - (a * a + b * b + c * c) 
        px = (a * (b + c - a) * x1 + b * (c + a - b) * x2 + c * (a + b - c) * x3) / s
        py = (a * (b + c - a) * y1 + b * (c + a - b) * y2 + c * (a + b - c) * y3) / s 
        ar = a ** 0.5
        br = b ** 0.5
        cr = c ** 0.5 
        r = ar * br * cr / ((ar + br + cr) * (-ar + br + cr) * (ar - br + cr) * (ar + br - cr)) ** 0.5
        return (np.array((px, py, 0)), r)

    @staticmethod
    def circle_angle(p, center):
        return math.atan2(p[1] - center[1], p[0] - center[0])

class CircularInversionIntro(ma.Scene):
    def setup(self) -> None:
        self.RADIUS = 2
        self.inversion_circle = ma.Circle(radius=self.RADIUS, color=ma.BLUE)

        self.origin = ManimUtils.create_point(self.inversion_circle.get_center(), r"O")
        self.A = ManimUtils.create_point(np.array((1, 0, 0)), r"A", color=ma.GREEN)
        self.A_prime = ManimUtils.create_point(np.zeros((3,)), r"A'", color=ma.RED)

        # origin-A / origin-A' line
        self.line = ma.Line(self.origin["dot"], self.A_prime["dot"])
        self.radius_line = ma.Line(self.origin["dot"], np.array((0, -self.RADIUS, 0)))
        self.radius_label = ma.Tex("R = " + str(self.RADIUS)).align_to(self.origin["dot"]).shift(ma.LEFT / 3, ma.DOWN *self.RADIUS / 2).rotate(ma.TAU / 4)
        self.total = ma.VGroup(
            self.inversion_circle,
            self.origin,
            self.A,
            self.radius_line,
            self.radius_label
        )

        # Constrain Eq
        oa = r"\overline{OA}"
        oa_prime = r"\overline{OA'}"
        self.constraint = (
            ma.MathTex(oa, r"\cdot", oa_prime, r"= R^2")
            .set_color_by_tex(oa, self.A.color)
            .set_color_by_tex(oa_prime, self.A_prime.color)
            .align_to(self.origin["dot"])
            .shift(ma.RIGHT * 4, ma.UP / 2)
        )

        self.ex = self._new_ex()

    def _new_ex(self) -> ma.MathTex:
        oa = "1"
        oa_prime = "4"

        return (
            ma.MathTex(r"\frac{2^2}{", oa, r"} = ", oa_prime)
            .set_color_by_tex(oa, self.A.color)
            .set_color_by_tex(oa_prime, self.A_prime.color)
            .align_to(self.origin["dot"])
            .shift(ma.RIGHT * 4, ma.DOWN / 1.5)
        )

    def rearrange_constraint(self) -> ma.MathTex:
        # Scuffed way to rearrange the constraint eq
        oa = r"\overline{OA}"
        oa_prime = r"\overline{OA'}"
        return (
            ma.MathTex(r"\frac{R^2}{", oa, r"} = ", oa_prime)
            .set_color_by_tex(oa, self.A.color)
            .set_color_by_tex(oa_prime, self.A_prime.color)
            .align_to(self.origin["dot"])
            .shift(ma.RIGHT * 4, ma.UP / 1.5)
        )

    def add_updaters(self) -> None:
        self.A_prime.add_updater(
            lambda x: x["dot"].move_to(
                MathUtils.circ_inverse_of(
                    self.A["dot"].get_center(),
                    self.origin["dot"].get_center(),
                    self.RADIUS,
                )
            )
            and x["txt"].move_to(x["dot"]).shift(ma.UP / 2),
        )
        self.line.add_updater(
            lambda x: x.become(
                ma.Line(
                    self.origin["dot"].get_center(),
                    max(
                        self.A["dot"],
                        self.A_prime["dot"],
                        key=lambda x: MathUtils.distance(
                            x.get_center(), self.origin["dot"].get_center()
                        ),
                    ),
                ),
                copy_submobjects=False,
            )
        )

    def trace_circle(self, r, c) -> None:
        self.circle = ma.Circle(radius=r, color=ma.GREEN).shift(c[0] * ma.RIGHT, c[1] * ma.UP)
        offset_circle = ma.Circle(radius=r, color=ma.GREEN).shift(c[0] * ma.RIGHT, (c[1] + 0.3) * ma.UP)
        inverted_circle_points = [MathUtils.circ_inverse_of(self.circle.get_center() + ma.RIGHT * self.circle.radius, self.origin["dot"].get_center(), self.RADIUS),
            MathUtils.circ_inverse_of(self.circle.get_center() + ma.UP * self.circle.radius, self.origin["dot"].get_center(), self.RADIUS),
            MathUtils.circ_inverse_of(self.circle.get_center() + ma.DOWN * self.circle.radius, self.origin["dot"].get_center(), self.RADIUS)
        ]
        center, radius = MathUtils.define_circle(inverted_circle_points[0], inverted_circle_points[1], inverted_circle_points[2])
        def inverted_arc_updater(obj):
            obj.become(
                    ma.Arc(
                        radius=radius,
                        arc_center=center,
                        start_angle=MathUtils.circle_angle(self.A_prime["dot"].get_center(), center),
                        angle=ma.PI - MathUtils.circle_angle(self.A_prime["dot"].get_center(), center),
                        color=ma.RED
                    ),
                    copy_submobjects=False
            )
        self.inverted_arc = ma.Arc(
            radius=radius,
            arc_center=center,
            start_angle=ma.PI,
            angle=0,
            color=ma.RED
        )
        self.inverted_arc.add_updater(inverted_arc_updater)
        self.play(ma.FadeInFromLarge(self.circle))
        self.wait(2)
        self.play(ma.FadeInFromLarge(self.inverted_arc))
        self.play(ma.MoveAlongPath(self.A, offset_circle), rate_func=ma.utils.rate_functions.smooth, run_time=8)
        self.inverted_arc.remove_updater(inverted_arc_updater)
        self.inverted_arc = ma.Circle(radius=radius, color=ma.RED).shift(center[0] * ma.RIGHT, center[1] * ma.UP)
        self.add(self.inverted_arc)
        self.wait(2)
        self.play(ma.Unwrite(self.inverted_arc))
        self.play(ma.Unwrite(self.circle))
        self.wait(2)

    def play_all(self) -> None:
        # Play animation
        self.play(ma.GrowFromCenter(self.inversion_circle))
        self.play(ma.FadeIn(self.origin))
        self.play(ma.FadeIn(self.radius_line))
        self.play(ma.FadeIn(self.radius_label))
        self.wait(2)
        self.play(ma.FadeInFromLarge(self.A))
        self.add_foreground_mobjects(self.A)
        self.wait(2)
        self.play(self.total.animate.shift(ma.LEFT * 3))

        self.play(ma.Write(self.constraint))
        self.wait(2)
        self.play(ma.Transform(self.constraint, self.rearrange_constraint()))
        self.play(ma.Write(self.ex))
        self.wait(2)
        self.play(ma.FadeIn(self.A_prime))
        self.play(ma.FadeIn(self.line))
        self.wait(2)

        self.play(ma.Unwrite(self.ex))
        self.play(ma.Unwrite(self.radius_line))
        self.play(ma.Unwrite(self.radius_label))
        self.play(self.constraint.animate.shift(ma.LEFT * 7, ma.DOWN * 3.5))
        self.wait(2)
        self.play(self.A.animate.shift(ma.UP), run_time=4)
        self.play(self.A.animate.shift(ma.DOWN), run_time=4)
        self.play(self.A.animate.shift(ma.RIGHT), run_time=4)
        self.wait(2)
        self.play(self.A.animate.shift(ma.RIGHT), run_time=4)
        self.wait(2)
        self.play(self.A.animate.shift(ma.LEFT * (3 - 0.001)), run_time=6)
        self.wait(2)
        self.play(self.A.animate.shift(ma.RIGHT * (1.5 - 0.001)), run_time=4)
        self.trace_circle(0.5, (-2, 0))
        self.play(self.A.animate.shift(ma.RIGHT * 1.5), run_time=4)
        self.trace_circle(1, (-1, 0))
        self.play(self.A.animate.shift(ma.LEFT * (2 - 0.001)), run_time=4)
        self.trace_circle(0.5, (-2.5 + 0.001, 0))
        

    def construct(self) -> None:
        self.setup()
        self.add_updaters()
        self.play_all()

# CircularInversionIntro().construct()
