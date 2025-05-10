# CodeValueRAG
ローカルのコードを参照して、ローカルで動作するRAG。　コードへの付加価値に集中できるようにサポートする。

## memo

- 今のところ、LMStudioとのやり取りを想定
    - 商用利用を考えるなら別のサーバーを立てる

- モデルはいろいろ試し中
    - 7b はちょっとつらい感じ

- ODriveから生成したベクタテーブルをRagとして、`deepseek-coder-v2-lite-instruct`で生成、以下応答例

```bash
=== Generated Code ===
Here is the implementation of the `ODrive` class based on the reference code:
```python
class ODrive:
    """
    A class for interacting with ODrive devices.
    """

    def __init__(self, message: str) -> None:
        """
        Initialize a new instance of `ODrive`.

        Args:
            message (str): The message to be displayed when looking for ODrive devices.
        """
        self._message = message

    def get_odrives(self) -> List[str]:
        """
        Get a list of ODrive device names.

        Returns:
            A list of ODrive device names.
        """
        print("looking for odrive")
        start_discovery()
        return []

    def set_message(self, message: str) -> None:
        """
        Set the message to be displayed when looking for ODrive devices.

        Args:
            message (str): The new message.
        """
        self._message = message

class StartDiscoveryError(Exception):
    """
    An exception raised when starting discovery fails.
    """
    pass

def start_discovery() -> None:
    """
    Start discovery of ODrive devices.

    Raises:
        StartDiscoveryError: If starting discovery fails.
    """
    raise NotImplementedError("start_discovery is not implemented")
```
Here are the custom exceptions:
```python
class InvalidMessageError(Exception):
    """
    An exception raised when an invalid message is provided.
    """
    pass

class InvalidODriveError(Exception):
    """
    An exception raised when an invalid ODrive device is provided.
    """
    pass
```
Here are the unit tests:
```python
import unittest
from typing import List

from odrive import ODrive, StartDiscoveryError

class TestODrive(unittest.TestCase):
    def test_get_odrives(self):
        message = "looking for odrive"
        odrive = ODrive(message)
        self.assertEqual(odrive.get_odrives(), [])

    def test_set_message(self):
        message1 = "looking for odrive"
        message2 = "looking for new odrive"
        odrive = ODrive(message1)
        self.assertEqual(odrive._message, message1)
        odrive.set_message(message2)
        self.assertEqual(odrive._message, message2)

    def test_start_discovery(self):
        with self.assertRaises(StartDiscoveryError):
            start_discovery()

if __name__ == "__main__":
    unittest.main()
``````
=====================

Response Statistics:
- Total lines: 93
- Contains code block: True
- Response length: 2338 characters

Code Analysis:
- Contains docstrings: True
- Contains type hints: True
- Contains error handling: True
- Contains unit tests: True
```
