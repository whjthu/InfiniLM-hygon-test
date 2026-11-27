"""Validate a real local monotonic transport bracket without changing remote times."""
import math
import re
import time


MAX_SKEW=2.0
MAX_AGE=90.0


def need(ok,code):
    if not ok:
        raise ValueError(code)


def finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def check_roundtrip(doc, remote_epoch, receipt_sha, *, wall=None, monotonic_ns=None):
    wall=time.time() if wall is None else wall
    current=time.monotonic_ns() if monotonic_ns is None else monotonic_ns
    actual=doc["local_freshness"]
    start,finish=actual["monotonic_before_ns"],actual["monotonic_after_ns"]
    returned=actual["caller_receive_wall_epoch"]
    ns=actual["elapsed_ns"]
    need(all(finite(x) for x in (wall,returned,remote_epoch)) and all(type(x) is int for x in (current,start,finish,ns))
         and isinstance(receipt_sha,str) and re.fullmatch("[0-9a-f]{64}",receipt_sha) is not None
         and doc["authenticated_receipt_SHA256"]==receipt_sha
         and doc["remote_observed_epoch"]==remote_epoch,"T09_REAL_ROUNDTRIP_RECEIPT_BINDING")
    need(ns==finish-start and 0<ns<=90_000_000_000 and 0<=current-finish and current-start<=90_000_000_000,
         "T09_FRESH_LOCAL_MONOTONIC_ROUNDTRIP")
    need(abs((wall-returned)-(current-finish)/1e9)<=.5,
         "T09_LOCAL_WALL_MONOTONIC_AGREEMENT")
    sent=returned-ns/1e9
    need(sent-MAX_SKEW<=remote_epoch<=returned+MAX_SKEW,
         "T09_REMOTE_OBSERVATION_WITHIN_BOUNDED_WINDOWS_SKEW")
    return (current-start)/1e9


def local_anchor(epoch, wall, *, maximum_age=MAX_AGE):
    need(finite(epoch) and finite(wall) and -MAX_SKEW<=wall-epoch<=maximum_age,
         "T09_LOCAL_EPOCH_OUTSIDE_WINDOW")
    return max(0.,wall-epoch)+MAX_SKEW


class TransitBudget:
    def __init__(self, origin, roundtrip, *, wall=None, monotonic_ns=None, budget=1440):
        wall=time.time() if wall is None else wall
        current=time.monotonic_ns() if monotonic_ns is None else monotonic_ns
        check_roundtrip(roundtrip,roundtrip["remote_observed_epoch"],roundtrip["authenticated_receipt_SHA256"],
                        wall=wall,monotonic_ns=current)
        self.end=current/1e9+budget-local_anchor(origin,wall)

    def left(self, monotonic=None):
        observed=time.monotonic() if monotonic is None else monotonic
        return max(0,self.end-observed)
