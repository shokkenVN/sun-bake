import qdarktheme

ADDITIONAL_QSS = """
QFrame#DropZone {
    border: 1px dashed #4b5563;
    border-radius: 6px;
}
QFrame#DropZone:hover {
    border-color: #60a5fa;
}
QPushButton#ActionBtn {
    font-weight: 600;
    padding: 8px 16px;
}
"""


def apply_theme(theme: str = "dark"):
    qdarktheme.setup_theme(
        theme=theme,
        corner_shape="rounded",
        additional_qss=ADDITIONAL_QSS,
    )
