import sys
import unittest
from types import ModuleType
from unittest.mock import patch

from smartpool.module_deps import module_deps


class ModuleDepsTests(unittest.TestCase):
    def test_does_not_inherit_unrelated_transitive_imports(self):
        parent = ModuleType("test_parent")
        direct = ModuleType("test_direct")
        unrelated = ModuleType("test_unrelated")
        direct.unrelated = unrelated
        parent.direct = direct

        with patch.dict(sys.modules, {
            parent.__name__: parent,
            direct.__name__: direct,
            unrelated.__name__: unrelated,
        }):
            deps = module_deps(parent)

        self.assertIn("test_parent", deps)
        self.assertIn("test_direct", deps)
        self.assertNotIn("test_unrelated", deps)
