import os
import shutil
from form.choice import Ui_Platform
import pandas as pd
from PyQt6.QtWidgets import (QMainWindow,
                             QFileDialog, QMessageBox)


def check_and_update_file_state(filename):
    states_file = "file_states.csv"

    # Создаем CSV файл состояний если его нет
    if not os.path.exists(states_file):
        df_states = pd.DataFrame(columns=['name', 'preprocessing', 'visualization', 'modeling'])
        df_states.to_csv(states_file, index=False)

    # Загружаем текущие состояния
    df_states = pd.read_csv(states_file)

    # Проверяем есть ли запись для этого файла
    if filename not in df_states['name'].values:
        # Добавляем новую запись
        new_row = pd.DataFrame({
            'name': [filename],
            'preprocessing': [False],
            'visualization': [False],
            'modeling': [False]
        })
        df_states = pd.concat([df_states, new_row], ignore_index=True)
        df_states.to_csv(states_file, index=False)
        print(f"Добавлена новая запись для файла: {filename}")
    else:
        # Получаем текущее состояние
        file_state = df_states[df_states['name'] == filename].iloc[0]
        print(f"Текущее состояние файла {filename}:")
        print(f"  Предобработка: {file_state['preprocessing']}")
        print(f"  Визуализация: {file_state['visualization']}")
        print(f"  Моделирование: {file_state['modeling']}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.data_folder = 'data/storage'
        self.states_file = "data/app_data/file_states.csv"
        self.current_filename = None

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
        self.ui.back_button.clicked.connect(self.go_back_to_main)
        self.ui.visualization_button.setEnabled(False)
        self.ui.modeling_button.setEnabled(False)
        self.ui.file_list.clearSelection()
        self.ui.add_button.setFocus()

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
            self.cleanup_orphaned_states()
            self.ui.file_list.setCurrentItem(None)

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
                self.check_and_update_file_state(filename)

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

        if not os.path.exists(file_path):
            QMessageBox.critical(self, "Ошибка", f"Файл {filename} не найден!")
            return
        self.current_filename = filename
        self.check_and_update_file_state(filename)
        self.update_analysis_buttons_state()
        self.ui.stackedWidget.setCurrentIndex(1)
        self.ui.label_4.setText(
            f"<h1 style='color: #1e3a5f; margin: 15px; text-align: center;'>🔧 Анализ: {filename}</h1>")
        QMessageBox.information(self, "Успех",
                                f"Файл {filename} успешно загружен для анализа!")

    def check_and_update_file_state(self, filename):
        """Проверяем и обновляем состояние файла в CSV состояний"""

        # Создаем CSV файл состояний если его нет
        if not os.path.exists(self.states_file):
            df_states = pd.DataFrame(columns=['name', 'preprocessing', 'visualization', 'modeling'])
            df_states.to_csv(self.states_file, index=False)

        # Загружаем текущие состояния
        df_states = pd.read_csv(self.states_file)

        # Проверяем есть ли запись для этого файла
        if filename not in df_states['name'].values:
            # Добавляем новую запись
            new_row = pd.DataFrame({
                'name': [filename],
                'preprocessing': [False],
                'visualization': [False],
                'modeling': [False]
            })
            df_states = pd.concat([df_states, new_row], ignore_index=True)
            df_states.to_csv(self.states_file, index=False)

    def update_analysis_buttons_state(self):
        if not self.current_filename:
            return

        if not os.path.exists(self.states_file):
            # Если файла нет, все кнопки кроме предобработки disabled
            self.set_buttons_state(True, False, False)
            return

        df_states = pd.read_csv(self.states_file)
        if self.current_filename in df_states['name'].values:
            file_state = df_states[df_states['name'] == self.current_filename].iloc[0]

            # Визуализация доступна только после предобработки
            visualization_enabled = bool(file_state['preprocessing'])
            # Моделирование доступно только после предобработки
            modeling_enabled = bool(file_state['preprocessing'])

            self.set_buttons_state(True, visualization_enabled, modeling_enabled)
        else:
            QMessageBox.critical(self, "Ошибка", f"ЧТО-ТО НЕ ТАК С ФАЙЛОМ СОСТОЯНИЯ")
            self.set_buttons_state(True, False, False)

    def set_buttons_state(self, preprocessing_enabled, visualization_enabled, modeling_enabled):
        """Устанавливает состояние кнопок анализа"""
        self.ui.preprocessing_button.setEnabled(preprocessing_enabled)
        self.ui.visualization_button.setEnabled(visualization_enabled)
        self.ui.modeling_button.setEnabled(modeling_enabled)

        # Визуально показываем состояние кнопок
        style_disabled = "background-color: #cccccc; color: #666666; border: none; border-radius: 15px; font-weight: bold; font-size: 22px; padding: 20px;"

        if not preprocessing_enabled:
            self.ui.preprocessing_button.setStyleSheet(style_disabled)
        else:
            self.ui.preprocessing_button.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                            stop: 0 #4CAF50, stop: 1 #45a049);
                    color: white;
                    border: none;
                    border-radius: 15px;
                    font-weight: bold;
                    font-size: 20px;
                    padding: 20px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                            stop: 0 #45a049, stop: 1 #3d8b40);
                }
                QPushButton:pressed {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                            stop: 0 #3d8b40, stop: 1 #357c38);
                }
            """)

        if not visualization_enabled:
            self.ui.visualization_button.setStyleSheet(style_disabled)
        else:
            self.ui.visualization_button.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                            stop: 0 #2196F3, stop: 1 #1976D2);
                    color: white;
                    border: none;
                    border-radius: 15px;
                    font-weight: bold;
                    font-size: 20px;
                    padding: 20px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                            stop: 0 #1976D2, stop: 1 #1565C0);
                }
                QPushButton:pressed {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                            stop: 0 #1565C0, stop: 1 #0D47A1);
                }
            """)

        if not modeling_enabled:
            self.ui.modeling_button.setStyleSheet(style_disabled)
        else:
            self.ui.modeling_button.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                            stop: 0 #FF9800, stop: 1 #F57C00);
                    color: white;
                    border: none;
                    border-radius: 15px;
                    font-weight: bold;
                    font-size: 20px;
                    padding: 20px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                            stop: 0 #F57C00, stop: 1 #EF6C00);
                }
                QPushButton:pressed {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                            stop: 0 #EF6C00, stop: 1 #E65100);
                }
            """)

    def update_file_state(self, step_name, completed=True):
        """Обновляет состояние конкретного шага для текущего файла"""
        if not self.current_filename:
            return
        if not os.path.exists(self.states_file):
            QMessageBox.critical(self, "Ошибка", f"ЧТО-ТО НЕ ТАК С ФАЙЛОМ СОСТОЯНИЯ")

        df_states = pd.read_csv(self.states_file)
        if self.current_filename in df_states['name'].values:
            # Обновляем состояние
            df_states.loc[df_states['name'] == self.current_filename, step_name] = completed
            df_states.to_csv(self.states_file, index=False)

    def go_back_to_main(self):
        self.ui.stackedWidget.setCurrentIndex(0)

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
                self.remove_file_state(filename)
                os.remove(file_path)
                self.update_file_list()
                QMessageBox.information(self, "Успех", f"Датасет '{filename}' удален!")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось удалить файл: {str(e)}")

    def remove_file_state(self, filename):
        """Удаляет запись о файле из CSV состояний"""

        if not os.path.exists(self.states_file):
            QMessageBox.critical(self, "Ошибка", f"ЧТО-ТО НЕ ТАК С ФАЙЛОМ СОСТОЯНИЯ")
            return

        try:
            df_states = pd.read_csv(self.states_file)

            # Проверяем есть ли запись для этого файла
            if filename in df_states['name'].values:
                # Удаляем строку с этим файлом
                df_states = df_states[df_states['name'] != filename]
                df_states.to_csv(self.states_file, index=False)
            else:
                QMessageBox.critical(self, "Ошибка", f"1ЧТО-ТО НЕ ТАК С ФАЙЛОМ СОСТОЯНИЯ")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"2ЧТО-ТО НЕ ТАК С ФАЙЛОМ СОСТОЯНИЯ")

    # Также нужно обновить функцию обновления списка файлов, чтобы удалять состояния несуществующих файлов
    def cleanup_orphaned_states(self):
        """Очищает состояния файлов, которых больше нет в папке данных"""
        if not os.path.exists(self.states_file):
            return
        df_states = pd.read_csv(self.states_file)
        existing_files = set(os.listdir(self.data_folder))
        # Находим файлы в состояниях, которых нет в папке
        orphaned_files = df_states[~df_states['name'].isin(existing_files)]

        if not orphaned_files.empty:
            # Удаляем orphaned записи
            df_states = df_states[df_states['name'].isin(existing_files)]
            df_states.to_csv(self.states_file, index=False)
