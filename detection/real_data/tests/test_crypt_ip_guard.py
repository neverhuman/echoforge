from __future__ import annotations

import io
import unittest

from detection.crypt_ip_guard import (
    DEVELOPER_KEY_REQUIRED_EXIT_CODE,
    DeveloperKeyRequiredError,
    raise_for_private_import_error,
    run_with_developer_key_warning,
)


class CryptIpGuardTests(unittest.TestCase):
    def test_missing_private_import_reports_unlock_instruction(self) -> None:
        with self.assertRaises(DeveloperKeyRequiredError) as caught:
            raise_for_private_import_error(
                "detection.crypt_ip_impl.__missing_for_guard_test__",
                ModuleNotFoundError(
                    "No module named 'detection.crypt_ip_impl.__missing_for_guard_test__'",
                    name="detection.crypt_ip_impl.__missing_for_guard_test__",
                ),
            )

        message = str(caught.exception)
        self.assertIn("git-crypt unlock", message)
        self.assertIn("developer key", message)
        self.assertIn("detection.crypt_ip_impl.__missing_for_guard_test__", message)

    def test_cli_guard_prints_one_line_warning(self) -> None:
        stderr = io.StringIO()

        def fail() -> None:
            raise DeveloperKeyRequiredError(
                "Engineered Intelligence runtime code is protected by git-crypt; "
                "run `git-crypt unlock` with the developer key, then retry this command."
            )

        code = run_with_developer_key_warning(fail, stream=stderr)

        self.assertEqual(code, DEVELOPER_KEY_REQUIRED_EXIT_CODE)
        warning = stderr.getvalue()
        self.assertEqual(warning.count("\n"), 1)
        self.assertIn("warning:", warning)
        self.assertIn("git-crypt unlock", warning)


if __name__ == "__main__":
    unittest.main()
