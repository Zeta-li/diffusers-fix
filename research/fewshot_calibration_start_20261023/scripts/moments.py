import json
from pathlib import Path

import numpy as np


ETAS = (1e-4, 1e-2, 1.0)
EPSILON = 1e-12
SCHEMA_VERSION = 1


def _samples(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError("Samples must have nonempty shape [tokens, channels]")
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite samples")
    return value


def _key(*indices):
    if any(type(i) is not int or i < 0 for i in indices):
        raise ValueError("Layer and time indices must be nonnegative Python integers")
    return ":".join(map(str, indices))


def _record_check(record, names, channels=None):
    if type(record["n"]) is not int or record["n"] < 1:
        raise ValueError("Moment token count must be a positive integer")
    arrays = [np.asarray(record[name], dtype=np.float64) for name in names]
    if any(a.ndim != 1 or a.size == 0 or not np.isfinite(a).all() for a in arrays):
        raise ValueError("Moment sums must be finite nonempty channel vectors")
    if any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("Moment channel shapes differ")
    if channels is not None and arrays[0].size != channels:
        raise ValueError("Moment channels differ from requested geometry")
    return arrays


class ImageMoments:
    def __init__(self, image_id, metadata=None):
        if not isinstance(image_id, str) or not image_id:
            raise ValueError("image_id must identify a source image")
        self.data = {"schema_version": SCHEMA_VERSION, "image_id": image_id,
                     "a": {}, "b": {}, "metadata": dict(metadata or {})}

    def _add_sums(self, route, key, n, sums):
        record = {"n": n, **sums}
        names = ("sum_h", "sum_h2") if route == "a" else ("sum_x", "sum_y", "sum_x2", "sum_xy", "sum_y2")
        arrays = _record_check(record, names)
        sums = {name: value.copy() for name, value in zip(names, arrays)}
        record = self.data[route].get(key)
        if record is None:
            self.data[route][key] = {"n": n, **sums}
        else:
            if record[names[0]].shape != arrays[0].shape:
                raise ValueError("Cannot change channel geometry while streaming")
            updated = {name: record[name] + value for name, value in sums.items()}
            if not all(np.isfinite(v).all() for v in updated.values()):
                raise ValueError("Moment accumulation overflow")
            record["n"] += n
            record.update(updated)
        return self

    def add_a_sums(self, layer, anchor, n, sum_h, sum_h2):
        return self._add_sums("a", _key(layer, anchor), n, {"sum_h": sum_h, "sum_h2": sum_h2})

    def add_b_sums(self, layer, target, anchor, n, sum_x, sum_y, sum_x2, sum_xy, sum_y2):
        if anchor > target:
            raise ValueError("A real anchor cannot lie after its target step")
        return self._add_sums("b", _key(layer, target, anchor), n,
                              {"sum_x": sum_x, "sum_y": sum_y, "sum_x2": sum_x2,
                               "sum_xy": sum_xy, "sum_y2": sum_y2})

    def _add(self, route, key, values):
        count = values[0].shape[0]
        if route == "a":
            h = values[0]
            sums = {"sum_h": h.sum(axis=0), "sum_h2": np.square(h).sum(axis=0)}
        else:
            x, y = values
            if x.shape != y.shape:
                raise ValueError("Old and fresh Value shapes differ")
            sums = {"sum_x": x.sum(axis=0), "sum_y": y.sum(axis=0),
                    "sum_x2": np.square(x).sum(axis=0), "sum_xy": (x*y).sum(axis=0),
                    "sum_y2": np.square(y).sum(axis=0)}
        if not all(np.isfinite(v).all() for v in sums.values()):
            raise ValueError("Moment accumulation overflow")
        return self._add_sums(route, key, count, sums)

    def add_a(self, layer, anchor, h):
        return self._add("a", _key(layer, anchor), [_samples(h)])

    def add_b(self, layer, target, anchor, x, y):
        if anchor > target:
            raise ValueError("A real anchor cannot lie after its target step")
        return self._add("b", _key(layer, target, anchor), [_samples(x), _samples(y)])

    def as_dict(self):
        return {"schema_version": SCHEMA_VERSION, "image_id": self.data["image_id"],
                "metadata": dict(self.data["metadata"]),
                **{route: {key: {name: value.copy() if isinstance(value, np.ndarray) else value
                                 for name, value in record.items()}
                           for key, record in self.data[route].items()} for route in ("a", "b")}}

    @classmethod
    def from_dict(cls, data):
        value = cls(data["image_id"], data.get("metadata"))
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported moments schema")
        for route, names in (("a", ("sum_h", "sum_h2")),
                             ("b", ("sum_x", "sum_y", "sum_x2", "sum_xy", "sum_y2"))):
            for key, record in data[route].items():
                arrays = _record_check(record, names)
                value.data[route][key] = {"n": record["n"], **dict(zip(names, (a.copy() for a in arrays)))}
        return value


def save_npz(path, data):
    if isinstance(data, ImageMoments):
        data = data.as_dict()
    arrays = {}

    def encode(value):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ValueError("Object arrays are not supported")
            key = f"array_{len(arrays):06d}"
            arrays[key] = value
            return {"type": "array", "key": key}
        if isinstance(value, dict):
            if not all(isinstance(k, str) for k in value):
                raise ValueError("Dictionary keys must be strings")
            return {"type": "dict", "items": {k: encode(v) for k, v in value.items()}}
        if isinstance(value, (list, tuple)):
            return {"type": "list", "items": [encode(v) for v in value]}
        if isinstance(value, np.generic):
            value = value.item()
        if value is None or isinstance(value, (str, int, float, bool)):
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError("Nonfinite metadata")
            return {"type": "scalar", "value": value}
        raise TypeError(f"Unsupported npz value type: {type(value)}")

    tree = encode(data)
    arrays["__tree__"] = np.asarray(json.dumps(tree, ensure_ascii=False, allow_nan=False))
    with Path(path).open("wb") as stream:
        np.savez_compressed(stream, **arrays)


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        tree = json.loads(str(archive["__tree__"].item()))

        def decode(node):
            if node["type"] == "array":
                return archive[node["key"]].copy()
            if node["type"] == "dict":
                return {k: decode(v) for k, v in node["items"].items()}
            if node["type"] == "list":
                return [decode(v) for v in node["items"]]
            if node["type"] == "scalar":
                return node["value"]
            raise ValueError("Unsupported npz tree node")

        return decode(tree)


def _images(images):
    values = [v.as_dict() if isinstance(v, ImageMoments) else v for v in images]
    ids = [v["image_id"] for v in values]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate source image IDs must be merged before equal-image fitting")
    if any(v.get("schema_version") != SCHEMA_VERSION for v in values):
        raise ValueError("Unsupported moments schema")
    return sorted(values, key=lambda v: v["image_id"])


def _geometry(layers, steps, channels):
    if any(type(v) is not int or v < 1 for v in (layers, steps, channels)):
        raise ValueError("Geometry must contain positive Python integers")


def _combined(images, route, names, layers, steps, channels):
    dimensions = (layers, steps) if route == "a" else (layers, steps, steps)
    means = {name: np.zeros(dimensions + (channels,), dtype=np.float64) for name in names}
    counts = np.zeros(dimensions, dtype=np.int64)
    tokens = np.zeros(dimensions, dtype=np.int64)
    for image in images:
        for key, record in image[route].items():
            try:
                index = tuple(map(int, key.split(":")))
            except (ValueError, AttributeError) as exc:
                raise ValueError("Invalid moment index") from exc
            if len(index) != len(dimensions) or any(i < 0 or i >= dim for i, dim in zip(index, dimensions)):
                raise ValueError("Moment index lies outside requested geometry")
            if route == "b" and index[2] > index[1]:
                raise ValueError("Observed anchor lies after target time")
            arrays = _record_check(record, names, channels)
            for name, value in zip(names, arrays):
                means[name][index] += value / record["n"]
            counts[index] += 1
            tokens[index] += record["n"]
    for name in names:
        np.divide(means[name], counts[..., None], out=means[name], where=counts[..., None] > 0)
    return means, counts, tokens


def _nonnegative(value):
    scale = np.maximum(1.0, np.abs(value))
    if np.any(value < -1e-9 * scale):
        raise ValueError("Inconsistent moments produce negative variance or SSE")
    return np.maximum(value, 0.0)


def fit_route_a(images, layers, steps, channels):
    _geometry(layers, steps, channels)
    images = _images(images)
    sums, count, tokens = _combined(images, "a", ("sum_h", "sum_h2"), layers, steps, channels)
    mean = sums["sum_h"]
    second = sums["sum_h2"]
    variance = _nonnegative(second - np.square(mean))
    observed = count > 0
    variance[~observed] = 1.0
    second[~observed] = 1.0
    return {"mean": mean, "variance": variance, "second_moment": second,
            "coverage": observed, "source_image_counts": count, "token_counts": tokens,
            "fallback": "unobserved anchor uses mean zero, variance one; coverage remains false",
            "weighting": "equal source images over raw first/second moments, then population variance",
            "source_image_ids": [v["image_id"] for v in images]}


def fit_route_b(images, layers, steps, channels, eta):
    _geometry(layers, steps, channels)
    if not np.isfinite(eta) or eta < 0:
        raise ValueError("eta must be finite and nonnegative")
    images = _images(images)
    sums, count, tokens = _combined(images, "b", ("sum_x", "sum_y", "sum_x2", "sum_xy", "sum_y2"),
                                     layers, steps, channels)
    ex, ey = sums["sum_x"], sums["sum_y"]
    var = _nonnegative(sums["sum_x2"] - ex*ex)
    cov = sums["sum_xy"] - ex*ey
    regularizer = eta * var.mean(axis=-1) + EPSILON
    alpha = (cov + regularizer[..., None]) / (var + regularizer[..., None])
    beta = ey - alpha*ex
    observed = count > 0
    causal = np.arange(steps)[:, None] > np.arange(steps)[None, :]
    coverage = observed & causal[None]
    alpha[~coverage] = 1.0
    beta[~coverage] = 0.0
    if not np.isfinite(alpha).all() or not np.isfinite(beta).all():
        raise ValueError("Nonfinite fitted coefficients")
    return {"alpha": alpha, "beta": beta, "coverage": coverage, "observed_moments": observed,
            "source_image_counts": count, "token_counts": tokens, "regularizer": regularizer,
            "var_x": var, "cov_xy": cov, "mean_x": ex, "mean_y": ey,
            "eta": float(eta), "epsilon": EPSILON,
            "fallback": "unobserved and same-time pairs use alpha=1,beta=0",
            "weighting": "equal source images over raw moments before variance and covariance",
            "source_image_ids": [v["image_id"] for v in images]}


def regression_error(image, coefficients):
    numerator, teacher_energy, scalar_count, fallback_pairs = 0.0, 0.0, 0, 0
    for key, record in image["b"].items():
        index = tuple(map(int, key.split(":")))
        if index[1] == index[2]:
            continue
        alpha, beta = coefficients["alpha"][index], coefficients["beta"][index]
        sx, sy, sx2, sxy, sy2 = _record_check(record, ("sum_x", "sum_y", "sum_x2", "sum_xy", "sum_y2"), alpha.size)
        sse = sy2 + alpha*alpha*sx2 + record["n"]*beta*beta - 2*alpha*sxy - 2*beta*sy + 2*alpha*beta*sx
        numerator += float(_nonnegative(sse).sum())
        teacher_energy += float(_nonnegative(sy2).sum())
        scalar_count += record["n"] * alpha.size
        fallback_pairs += int(not coefficients["coverage"][index])
    if scalar_count == 0:
        raise ValueError("LOO source image has no off-diagonal observed regression labels")
    return {"sse": numerator, "teacher_energy": teacher_energy,
            "normalized_sse": numerator / (teacher_energy + EPSILON),
            "scalar_count": scalar_count, "fallback_pair_count": fallback_pairs}


def select_eta_loo(images, layers, steps, channels):
    images = _images(images)
    if len(images) < 2:
        raise ValueError("Source-image LOO selection requires at least two source images")
    candidates = []
    for eta in ETAS:
        folds = []
        for heldout in images:
            training = [v for v in images if v["image_id"] != heldout["image_id"]]
            coefficients = fit_route_b(training, layers, steps, channels, eta)
            folds.append({"heldout_image_id": heldout["image_id"],
                          "training_image_ids": coefficients["source_image_ids"],
                          **regression_error(heldout, coefficients)})
        candidates.append({"eta": eta, "score": float(np.mean([f["normalized_sse"] for f in folds])), "folds": folds})
    selected = min(candidates, key=lambda v: (v["score"], v["eta"]))
    return {"selected_eta": selected["eta"], "candidate_scores": candidates,
            "epsilon": EPSILON, "candidates": list(ETAS), "tie_break": "smaller eta",
            "reduction": "sum SSE and teacher raw second-moment energy over observed off-diagonal pairs/channels/tokens inside each heldout image; divide once per image; mean source-image scores",
            "source_image_ids": [v["image_id"] for v in images], "new_model_queries": 0}


def fit_calibration(images, layers, steps, channels):
    images = _images(images)
    selection = select_eta_loo(images, layers, steps, channels)
    return {"schema_version": SCHEMA_VERSION, "geometry": {"layers": layers, "steps": steps, "channels": channels},
            "route_a": fit_route_a(images, layers, steps, channels),
            "route_b": fit_route_b(images, layers, steps, channels, selection["selected_eta"]),
            "loo": selection, "source_image_ids": [v["image_id"] for v in images],
            "source_image_count": len(images), "new_model_queries": 0,
            "status": "offline_moment_fit_only; no deployment or image-quality claim"}


def deployment_route_b(calibration, heads, head_dim, contract, source, source_hash):
    geometry = calibration["geometry"]
    if geometry["steps"] != 8 or geometry["channels"] != heads*head_dim:
        raise ValueError("Deployment requires eight steps and matching head geometry")
    if not isinstance(contract, dict) or not isinstance(source, str) or not isinstance(source_hash, str):
        raise ValueError("Deployment contract and source provenance are required")
    shape = (geometry["layers"], 8, 8, heads, head_dim)
    route = calibration["route_b"]
    return {"alpha": route["alpha"].reshape(shape).astype(np.float32),
            "delta": route["beta"].reshape(shape).astype(np.float32),
            "observed_pairs": route["coverage"].copy(), "contract": dict(contract),
            "source": source, "source_hash": source_hash,
            "training_image_ids": list(calibration["source_image_ids"]),
            "eta": route["eta"], "epsilon": EPSILON,
            "status": "fitted coefficients; deployment validation remains caller responsibility"}
