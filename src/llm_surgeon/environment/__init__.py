from __future__ import annotations

from .base import EnvironmentProfile
from .detect import HostProbe
from .lambda_stack import LambdaStackProfile
from .ubuntu import UbuntuProfile


def get_profile(name: str, probe: HostProbe) -> EnvironmentProfile:
    return {"lambda-stack": LambdaStackProfile, "ubuntu": UbuntuProfile}[name](probe)
