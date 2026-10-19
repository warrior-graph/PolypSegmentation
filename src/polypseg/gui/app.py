"""Ponto de entrada da interface: ``polypseg-gui [imagem-ou-pasta]``."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from polypseg.gui.window import MainWindow


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    app = QApplication(argv)
    app.setApplicationName("PolypSeg")
    app.setStyle("Fusion")  # segue o tema claro/escuro do sistema no Qt 6
    window = MainWindow()
    window.show()
    if len(argv) > 1:
        target = Path(argv[1])
        if target.is_dir():
            window.open_folder(target)
        elif target.is_file():
            window.load_image(target)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
