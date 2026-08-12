# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.

import pytest

import script_guard


@pytest.mark.parametrize(
    "source",
    [
        "import os",
        "open('/tmp/x', 'w')",
        "torch.hub.load('owner/repo', 'model')",
        "torch.save({'x': 1}, '/tmp/x.pt')",
        "torch.os.remove('/tmp/x')",
        "value.__class__",
        "from torch import _C",
        "import torch.hub",
        "import torch as framework\nframework.hub.load('x', 'y')",
        "from torch import distributed",
        "torch.from_file('/var/lib/computefield-machine/identity.json')",
        "tensor.numpy().tofile('/tmp/output')",
        "tensor.dump('/tmp/output')",
    ],
)
def test_policy_rejects_escape_primitives(source):
    with pytest.raises(script_guard.UnsafeScript):
        script_guard.validate_script(source)


def test_safe_namespace_has_no_file_or_dynamic_code_builtins():
    namespace = {}
    script_guard.execute_script("answer = sum(range(5))", namespace)
    assert namespace["answer"] == 10
    assert "open" not in namespace["__builtins__"]
    assert "eval" not in namespace["__builtins__"]
