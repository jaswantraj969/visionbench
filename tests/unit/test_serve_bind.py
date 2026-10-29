"""`visionbench serve` binds to loopback only; --allow-non-loopback works in the container only."""

from visionbench.cli import bind_refusal, build_parser


def test_loopback_hosts_are_always_allowed() -> None:
    for host in ("127.0.0.1", "localhost", "::1"):
        assert bind_refusal(host, False, {}) is None


def test_non_loopback_is_refused_without_the_flag_even_in_the_container() -> None:
    reason = bind_refusal("0.0.0.0", False, {"VISIONBENCH_CONTAINER": "1"})
    assert reason is not None and "no authentication" in reason


def test_the_flag_alone_is_refused_outside_the_container() -> None:
    for env in ({}, {"VISIONBENCH_CONTAINER": "0"}, {"VISIONBENCH_CONTAINER": "true"}):
        reason = bind_refusal("0.0.0.0", True, env)
        assert reason is not None and "container image" in reason


def test_the_flag_inside_the_container_allows_any_host() -> None:
    assert bind_refusal("0.0.0.0", True, {"VISIONBENCH_CONTAINER": "1"}) is None


def test_the_flag_is_off_by_default() -> None:
    args = build_parser().parse_args(["serve"])
    assert args.allow_non_loopback is False
    assert build_parser().parse_args(["serve", "--allow-non-loopback"]).allow_non_loopback
