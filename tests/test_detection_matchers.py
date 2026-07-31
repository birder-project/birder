import math
import unittest

import torch

from birder.net.detection.hungarian_matcher import Assignment
from birder.net.detection.hungarian_matcher import HungarianMatcher


class TestHungarianMatcher(unittest.TestCase):
    @staticmethod
    def _assert_assignments_equal(actual: list[Assignment], expected: list[Assignment]) -> None:
        if len(actual) != len(expected):
            raise AssertionError(f"Assignment lengths differ: {len(actual)} != {len(expected)}")

        for actual_assignment, expected_assignment in zip(actual, expected, strict=True):
            for actual_indices, expected_indices in zip(actual_assignment, expected_assignment, strict=True):
                torch.testing.assert_close(actual_indices, expected_indices, rtol=0, atol=0)

    def test_class_cost(self) -> None:
        class_logits = torch.tensor(
            (
                ((-4.0, -4.0, 8.0), (8.0, -4.0, -4.0), (-4.0, 8.0, -4.0)),
                ((8.0, -4.0, -4.0), (-4.0, -4.0, 8.0), (-4.0, 8.0, -4.0)),
            )
        )
        box_predictions = torch.tensor((0.5, 0.5, 0.2, 0.2)).expand(2, 3, 4).clone()
        targets = [
            {
                "labels": torch.tensor((0, 2)),
                "boxes": torch.tensor((0.5, 0.5, 0.2, 0.2)).expand(2, 4).clone(),
            },
            {"labels": torch.tensor((1,)), "boxes": torch.tensor(((0.5, 0.5, 0.2, 0.2),))},
        ]
        expected = [
            (torch.tensor((0, 1)), torch.tensor((1, 0))),
            (torch.tensor((2,)), torch.tensor((0,))),
        ]

        for use_focal_cost in (False, True):
            with self.subTest(use_focal_cost=use_focal_cost):
                matcher = HungarianMatcher(
                    class_weight=1.0, bbox_weight=0.0, giou_weight=0.0, use_focal_cost=use_focal_cost
                )
                self._assert_assignments_equal(matcher.match(class_logits, box_predictions, targets), expected)

    def test_box_costs(self) -> None:
        class_logits = torch.zeros((1, 3, 1))
        box_predictions = torch.tensor((((0.8, 0.8, 0.1, 0.1), (0.2, 0.2, 0.1, 0.1), (0.5, 0.5, 0.1, 0.1)),))
        targets = [
            {
                "labels": torch.tensor((0, 0)),
                "boxes": torch.tensor(((0.2, 0.2, 0.1, 0.1), (0.8, 0.8, 0.1, 0.1))),
            }
        ]
        expected = [(torch.tensor((0, 1)), torch.tensor((1, 0)))]

        configurations = ((1.0, 0.0, True), (0.0, 1.0, True), (0.0, 1.0, False))
        for bbox_weight, giou_weight, use_giou in configurations:
            with self.subTest(bbox_weight=bbox_weight, giou_weight=giou_weight, use_giou=use_giou):
                matcher = HungarianMatcher(
                    class_weight=0.0, bbox_weight=bbox_weight, giou_weight=giou_weight, use_giou=use_giou
                )
                self._assert_assignments_equal(matcher.match(class_logits, box_predictions, targets), expected)

    def test_grouped_matches_individual_and_chunked_matching(self) -> None:
        generator = torch.Generator().manual_seed(0)
        batch_size = 4
        num_groups = 5
        num_queries = 6
        num_classes = 4
        class_logits = torch.randn(
            (batch_size, num_groups, num_queries, num_classes), generator=generator, requires_grad=True
        )
        box_predictions = torch.rand((batch_size, num_groups, num_queries, 4), generator=generator)
        box_predictions[..., 2:].mul_(0.4).add_(0.05)

        targets = []
        for target_count in (2, 0, 3, 2):
            target_boxes = torch.rand((target_count, 4), generator=generator)
            target_boxes[..., 2:].mul_(0.4).add_(0.05)
            targets.append(
                {
                    "labels": torch.randint(num_classes, (target_count,), generator=generator),
                    "boxes": target_boxes,
                }
            )

        matcher = HungarianMatcher(class_weight=2.0, bbox_weight=5.0, giou_weight=2.0, clamp_predicted_box_sizes=True)
        expected = [
            matcher.match(class_logits[:, group_index], box_predictions[:, group_index], targets)
            for group_index in range(num_groups)
        ]

        for group_chunk_size in (None, 1, 2, 3):
            with self.subTest(group_chunk_size=group_chunk_size):
                actual = matcher.match_grouped(
                    class_logits, box_predictions, targets, group_chunk_size=group_chunk_size
                )
                self.assertEqual(len(actual), num_groups)
                for actual_group, expected_group in zip(actual, expected, strict=True):
                    self._assert_assignments_equal(actual_group, expected_group)

        self.assertIsNone(class_logits.grad)

    def test_empty_targets_and_more_targets_than_queries(self) -> None:
        matcher = HungarianMatcher(class_weight=0.0, bbox_weight=1.0, giou_weight=0.0)
        class_logits = torch.zeros((2, 2, 1))
        box_predictions = torch.tensor(
            (
                ((0.3, 0.3, 0.1, 0.1), (0.7, 0.7, 0.1, 0.1)),
                ((0.8, 0.8, 0.1, 0.1), (0.2, 0.2, 0.1, 0.1)),
            )
        )
        targets = [
            {"labels": torch.empty((0,), dtype=torch.int64), "boxes": torch.empty((0, 4))},
            {
                "labels": torch.tensor((0, 0, 0)),
                "boxes": torch.tensor(((0.2, 0.2, 0.1, 0.1), (0.5, 0.5, 0.1, 0.1), (0.8, 0.8, 0.1, 0.1))),
            },
        ]

        actual = matcher.match(class_logits, box_predictions, targets)
        expected = [
            (torch.empty((0,), dtype=torch.int64), torch.empty((0,), dtype=torch.int64)),
            (torch.tensor((0, 1)), torch.tensor((2, 0))),
        ]
        self._assert_assignments_equal(actual, expected)

    def test_replace_non_finite_costs(self) -> None:
        cost = torch.tensor(
            (
                ((1.0, 2.0), (3.0, 4.0)),
                ((-100.0, math.nan), (math.inf, 100.0)),
                ((math.nan, math.inf), (-math.inf, math.nan)),
            )
        )
        finite = torch.isfinite(cost)

        replaced = HungarianMatcher._replace_non_finite_costs(cost)

        self.assertTrue(torch.isfinite(replaced).all())
        torch.testing.assert_close(replaced[finite], cost[finite], rtol=0, atol=0)
        self.assertGreater(replaced[1, 0, 1].item(), 300.0)
        self.assertEqual(torch.unique(replaced[2]).numel(), 1)

    def test_input_validation(self) -> None:
        for weights in ((-1.0, 1.0, 1.0), (math.inf, 1.0, 1.0), (0.0, 0.0, 0.0)):
            with self.subTest(weights=weights):
                with self.assertRaises(ValueError):
                    HungarianMatcher(*weights)

        matcher = HungarianMatcher(1.0, 1.0, 1.0)
        targets = [{"labels": torch.tensor((0,)), "boxes": torch.rand((1, 4))}]
        with self.assertRaises(ValueError):
            matcher.match_grouped(torch.rand((1, 2, 3)), torch.rand((1, 2, 4)), targets)
