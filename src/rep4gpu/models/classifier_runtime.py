"""CUDA runtime for one-logit classifiers, including live inference graphs."""
from .phase_graph import PhaseGraphParty
from .text_classifier import linear_labels


class ClassifierParty(PhaseGraphParty):
    """Use exact modular matvec while retaining the device protocol checks."""

    def matmul(self, x, y):
        from ..gpu.matmul.classifier_matvec import local_products_classifier
        if x.binary or y.binary:
            raise ValueError('classifier matvec requires arithmetic shares')
        local = local_products_classifier(*x.parts, *y.parts, bits=self.bits)
        contributions = [None] * 5
        contributions[(self.party - 1) % 4], contributions[self.party], contributions[4] = local
        return self._reshare_product(contributions, False, x.device)


def prepare_classifier(rt, example_features, weights):
    """Capture fixed-shape label inference with checked live communication.

    Return a plan whose ``run(features, weights)`` consumes fresh nonce
    reservations and returns checked public labels. Preparation uses no live
    transport; input sharing remains outside the plan. Keep the runtime and
    plan for the session so mask counters never restart.
    """
    return rt.prepare_inference(lambda x, w: rt.open(linear_labels(rt, x, w)),
                                example_features, weights, check_at_send=True)
