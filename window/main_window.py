import sys
import os
import shutil
from form.choice import Ui_Platform
from PyQt6.QtWidgets import (QMainWindow,
                             QFileDialog, QMessageBox)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.data_folder = 'data/app_data'

        # Создаем экземпляр UI класса
        self.ui = Ui_Platform()
        # Применяем UI к нашему окну
        self.ui.setupUi(self)

        # Добавляем свою логику
        self.setup_custom_logic()

    def setup_custom_logic(self):
        self.update_file_list()
        self.ui.add_button.clicked.connect(self.add_file)
        self.ui.update_button.clicked.connect(self.update_file_list)
        self.ui.delete_button.clicked.connect(self.delete_file)
        self.ui.open_button.clicked.connect(self.open_file)

    def update_file_stats(self):
        """Обновление статистики файлов"""
        try:
            files = [f for f in os.listdir(self.data_folder) if f.endswith(('.csv', '.json'))]
            csv_count = sum(1 for f in files if f.endswith('.csv'))
            json_count = sum(1 for f in files if f.endswith('.json'))

            stats_text = f"📊 Статистика: {len(files)} файлов (CSV: {csv_count}, JSON: {json_count})"
            self.ui.stats_label.setText(stats_text)

        except FileNotFoundError:
            self.ui.stats_label.setText("❌ Папка не найдена")

    def update_file_list(self):
        """Обновление списка файлов"""
        self.ui.file_list.clear()
        try:
            files = os.listdir(self.data_folder)
            for file in files:
                if file.endswith(('.csv', '.json')):
                    # Добавляем иконку в зависимости от типа файла
                    icon = "📝" if file.endswith('.csv') else "{}"
                    self.ui.file_list.addItem(f"{icon} {file}")

            self.update_file_stats()

        except FileNotFoundError:
            QMessageBox.warning(self, "Ошибка", f"Папка {self.data_folder} не найдена!")

    def add_file(self):
        """Добавление нового файла"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите CSV или JSON файл",
            "",
            "Data Files (*.csv *.json);;All Files (*)"
        )

        if file_path:
            try:
                if not file_path.lower().endswith(('.csv', '.json')):
                    QMessageBox.warning(self, "Ошибка", "Пожалуйста, выберите CSV или JSON файл!")
                    return

                filename = os.path.basename(file_path)
                destination = os.path.join(self.data_folder, filename)

                shutil.copy2(file_path, destination)
                self.update_file_list()

                QMessageBox.information(self, "Успех", f"Датасет '{filename}' успешно добавлен!")

            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось добавить файл: {str(e)}")

    def open_file(self):
        """Открытие выбранного файла"""
        current_item = self.ui.file_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Предупреждение", "Выберите файл для анализа!")
            return

        filename = current_item.text().split(" ", 1)[1]
        file_path = os.path.join(self.data_folder, filename)
        pass

    def delete_file(self):
        """Удаление выбранного файла"""
        current_item = self.ui.file_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Предупреждение", "Выберите файл для удаления!")
            return

        filename = current_item.text().split(" ", 1)[1]
        file_path = os.path.join(self.data_folder, filename)

        reply = QMessageBox.question(
            self,
            "Подтверждение удаления",
            f"🗑️ Вы уверены, что хотите удалить датасет '{filename}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                os.remove(file_path)
                self.update_file_list()
                QMessageBox.information(self, "Успех", f"Датасет '{filename}' удален!")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось удалить файл: {str(e)}")
