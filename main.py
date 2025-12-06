import sys
import os
from PyQt6.QtWidgets import QApplication
from window.main_window import MainWindow


def setup_project():
    """Настройка проекта перед запуском"""
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Создаем структуру папок
    folders = [
        ('data', 'storage'),
        ('data', 'app_data'),
    ]

    for folder_parts in folders:
        folder_path = os.path.join(project_root, *folder_parts)
        os.makedirs(folder_path, exist_ok=True)

    return project_root


if __name__ == "__main__":
    # Настраиваем проект
    setup_project()

    # Запускаем приложение
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.setQuitOnLastWindowClosed(False)
    sys.exit(app.exec())
