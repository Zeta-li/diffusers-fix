import json
import tempfile
from pathlib import Path

import numpy as np

from moments import (EPSILON, ETAS, ImageMoments, deployment_route_b, fit_calibration, fit_route_a,
                     fit_route_b, load_npz, regression_error, save_npz, select_eta_loo)


def image(name, x, y=None, h=None, target=1, anchor=0):
    value = ImageMoments(name)
    if h is not None:
        value.add_a(0, anchor, np.asarray(h, dtype=np.float64))
    if y is not None:
        value.add_b(0, target, anchor, np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    return value.as_dict()


def same_tree(a, b):
    if isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            same_tree(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            same_tree(left, right)
    else:
        assert a == b


def main():
    tests = []
    x = np.arange(18, dtype=np.float64).reshape(6, 3) / 7
    y = 2.5*x + np.array([1.0, -3.0, 2.0])
    full = ImageMoments("stream").add_a(0, 0, x).add_b(0, 1, 0, x, y)
    chunks = ImageMoments("stream").add_a(0, 0, x[:2]).add_a(0, 0, x[2:])
    chunks.add_b(0, 1, 0, x[:2], y[:2]).add_b(0, 1, 0, x[2:], y[2:])
    for route in ("a", "b"):
        for key, record in full.as_dict()[route].items():
            for field, expected in record.items():
                np.testing.assert_allclose(chunks.as_dict()[route][key][field], expected, rtol=1e-14, atol=1e-14)
    imported = ImageMoments("stream")
    ar = full.as_dict()["a"]["0:0"]
    br = full.as_dict()["b"]["0:1:0"]
    imported.add_a_sums(0,0,**ar).add_b_sums(0,1,0,**br)
    same_tree(imported.as_dict(),full.as_dict())
    tests.append("streamed chunks equal batch raw moments")
    images = [image("one", [[0.]], [[0.]], [[0.]]),
              image("many", np.full((9,1), 10.), np.full((9,1),30.), np.full((9,1),10.))]
    a = fit_route_a(images, 1, 3, 1)
    b = fit_route_b(images, 1, 3, 1, 1e-4)
    np.testing.assert_allclose(a["mean"][0,0], [5.])
    np.testing.assert_allclose(a["variance"][0,0], [25.])
    np.testing.assert_allclose(b["var_x"][0,1,0], [25.])
    np.testing.assert_allclose(b["cov_xy"][0,1,0], [75.])
    assert a["token_counts"][0,0] == 10 and a["source_image_counts"][0,0] == 2
    tests.append("different token counts remain source-image equal; between-image means retained")
    images = [image("left", x[:3], y[:3]), image("right", x[3:], y[3:])]
    fit = fit_route_b(images, 1, 3, 3, 1e-4)
    var = fit["var_x"][0,1,0]
    lam = 1e-4*var.mean()+EPSILON
    expected = (2.5*var+lam)/(var+lam)
    np.testing.assert_allclose(fit["alpha"][0,1,0], expected, atol=1e-13)
    np.testing.assert_allclose(fit["beta"][0,1,0], y.mean(0)-expected*x.mean(0), atol=1e-13)
    almost = fit_route_b(images, 1, 3, 3, 0.)
    np.testing.assert_allclose(almost["alpha"][0,1,0], 2.5, atol=2e-11)
    np.testing.assert_allclose(almost["beta"][0,1,0], [1.,-3.,2.], atol=2e-11)
    tests.append("known affine relation recovers analytic ridge and unregularized limit")
    low = fit_route_b(images,1,3,3,1e-4)["alpha"][0,1,0]
    high = fit_route_b(images,1,3,3,1.)["alpha"][0,1,0]
    assert np.all(abs(high-1) < abs(low-1))
    below_one = [image("left",x[:3],.5*x[:3]),image("right",x[3:],.5*x[3:])]
    low2 = fit_route_b(below_one,1,3,3,1e-4)["alpha"][0,1,0]
    high2 = fit_route_b(below_one,1,3,3,1.)["alpha"][0,1,0]
    assert np.all(high2 > low2)
    tests.append("identity ridge shrinks toward one, not toward zero")
    constant = [image("a", [[2.],[2.]], [[7.],[9.]]), image("b", [[2.]], [[10.]])]
    fit = fit_route_b(constant,1,3,1,1.)
    assert fit["alpha"][0,1,0,0] == 1.
    np.testing.assert_allclose(fit["beta"][0,1,0], [7.])
    tests.append("constant old Value remains finite with intercept and epsilon floor")
    fit = fit_route_b(images,1,3,3,1e-4)
    for time in range(3):
        assert np.array_equal(fit["alpha"][0,time,time], np.ones(3))
        assert np.array_equal(fit["beta"][0,time,time], np.zeros(3))
    assert not fit["coverage"][0,2,0]
    assert np.array_equal(fit["alpha"][0,2,0], np.ones(3))
    assert np.array_equal(fit["beta"][0,2,0], np.zeros(3))
    a = fit_route_a(images,1,3,3)
    assert not a["coverage"].any() and np.array_equal(a["variance"], np.ones((1,3,3)))
    tests.append("same-time and missing B pairs identity; missing A anchors explicit prior and mask")
    observed_diagonal = [image("a", x, -3*x+7,target=0,anchor=0),
                         image("b", x, -3*x+7,target=0,anchor=0)]
    d = fit_route_b(observed_diagonal,1,3,3,1.)
    assert d["observed_moments"][0,0,0] and not d["coverage"][0,0,0]
    assert np.array_equal(d["alpha"][0,0,0], np.ones(3))
    tests.append("observed diagonal still forced identity")
    data = [image("a", [[-1.],[0.],[1.]], [[-2.],[0.],[2.]]),
            image("b", [[-2.],[0.],[2.]], [[-4.],[0.],[4.]]),
            image("outlier", [[-3.],[0.],[3.]], [[30.],[0.],[-30.]])]
    loo = select_eta_loo(data,1,3,1)
    assert loo["candidates"] == list(ETAS) and loo["new_model_queries"] == 0
    for candidate in loo["candidate_scores"]:
        manual = []
        for fold in candidate["folds"]:
            assert fold["heldout_image_id"] not in fold["training_image_ids"]
            held = next(i for i in data if i["image_id"] == fold["heldout_image_id"])
            train = [i for i in data if i is not held]
            coeff = fit_route_b(train,1,3,1,candidate["eta"])
            raw = held["b"]["0:1:0"]
            xv = {"a": [-1.,0.,1.], "b": [-2.,0.,2.], "outlier": [-3.,0.,3.]}[held["image_id"]]
            yv = {"a": [-2.,0.,2.], "b": [-4.,0.,4.], "outlier": [30.,0.,-30.]}[held["image_id"]]
            prediction = coeff["alpha"][0,1,0,0]*np.array(xv)+coeff["beta"][0,1,0,0]
            expected = np.square(np.array(yv)-prediction).sum()/(np.square(yv).sum()+EPSILON)
            np.testing.assert_allclose(fold["normalized_sse"],expected,rtol=1e-13,atol=1e-13)
            manual.append(expected)
        np.testing.assert_allclose(candidate["score"],np.mean(manual),rtol=1e-13)
    full_fit = fit_route_b(data,1,3,1,ETAS[0])
    leaked = np.mean([regression_error(i,full_fit)["normalized_sse"] for i in data])
    assert abs(leaked-loo["candidate_scores"][0]["score"]) > .1
    tests.append("LOO excludes entire heldout source; SSE matches direct residuals; no in-sample leakage")
    repeated = [image("a", [[-1.],[1.]], [[-2.],[2.]]), image("b", [[-1.],[1.]], [[-2.],[2.]])]
    many = [repeated[0],image("b", np.tile([[-1.],[1.]],(20,1)), np.tile([[-2.],[2.]],(20,1)))]
    np.testing.assert_allclose(select_eta_loo(repeated,1,3,1)["candidate_scores"][0]["score"],
                               select_eta_loo(many,1,3,1)["candidate_scores"][0]["score"],rtol=1e-11,atol=1e-14)
    tests.append("per-image energy normalization is invariant to repeated token samples")
    disjoint = [image("a", [[-1.],[1.]], [[-2.],[2.]],target=1),
                image("b", [[-1.],[1.]], [[-3.],[3.]],target=2)]
    assert all(f["fallback_pair_count"] == 1 for c in select_eta_loo(disjoint,1,3,1)["candidate_scores"] for f in c["folds"])
    tests.append("LOO missing-training pair evaluates preregistered identity fallback")
    result = fit_calibration(data,1,3,1)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/"image.npz"
        save_npz(path,full)
        same_tree(full.as_dict(),load_npz(path))
        save_npz(path,result)
        same_tree(result,load_npz(path))
        same_tree(full.as_dict(),ImageMoments.from_dict(full.as_dict()).as_dict())
    tests.append("safe no-pickle npz and dictionary roundtrips for moments and fitted arrays")
    deploy_fit = fit_calibration(data,1,8,1)
    artifact = deployment_route_b(deploy_fit,1,1,{"frozen":"test"},"test","test_sha")
    assert artifact["alpha"].shape == (1,8,8,1,1) and artifact["alpha"].dtype == np.float32
    assert artifact["delta"].shape == artifact["alpha"].shape
    assert artifact["observed_pairs"].shape == (1,8,8)
    assert artifact["training_image_ids"] == ["a","b","outlier"]
    tests.append("eight-step B deployment mapping beta to delta preserves coverage and source IDs")
    for bad in (data+[data[0]],):
        try:
            fit_calibration(bad,1,3,1)
        except ValueError:
            pass
        else:
            raise AssertionError("Duplicate source accepted")
    try:
        ImageMoments("bad").add_b(0,0,1,np.ones((2,1)),np.ones((2,1)))
    except ValueError:
        pass
    else:
        raise AssertionError("Future anchor accepted")
    tests.append("duplicate sources and impossible future anchors rejected")
    print(json.dumps({"status":"PASS","checks":len(tests),"tests":tests,
                      "eta_candidates":list(ETAS),"epsilon":EPSILON,
                      "model_calls":0,"dit_forwards":0,"device":"CPU numpy float64",
                      "scope":"synthetic engineering checks; no real Train8 acquisition or quality claim"},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
