import os
import sys
import shutil
import pandas as pd
from PyQt6.QtWidgets import (QMainWindow, QFileDialog,
                             QMessageBox, QListWidgetItem)
from PyQt6.QtCore import Qt

# Импортируем UI главного окна
try:
    from form.choice import Ui_Platform
except ImportError:
    print("Ошибка: Не удалось импортировать Ui_Platform из form.choice")
    sys.exit(1)

# Импортируем окно препроцессинга
try:
    from preprocessing_window import PreprocessingWindow
except ImportError:
    from window.preprocessing_window import PreprocessingWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.data_folder = 'data/storage'
        self.states_file = "data/app_data/file_states.csv"
        self.current_filename = None
        self.preprocessing_window = None  # Ссылка на окно препроцессинга

        # Создаем экземпляр UI класса
        self.ui = Ui_Platform()
        # Применяем UI к нашему окну
        self.ui.setupUi(self)

        # Добавляем свою логику
        self.setup_custom_logic()

        # Создаем необходимые папки если их нет
        self.create_necessary_folders()

    def create_necessary_folders(self):
        """Создает необходимые папки если они не существуют"""
        # Определяем директорию, где находится mainwindow.py
        current_file_dir = os.path.dirname(os.path.abspath(__file__))

        # Поднимаемся на один уровень вверх (в корень проекта)
        project_root = os.path.dirname(current_file_dir)

        # Создаем папки в корне проекта
        data_folder_abs = os.path.join(project_root, self.data_folder)
        states_file_abs = os.path.join(project_root, self.states_file)

        # Создаем папки
        os.makedirs(data_folder_abs, exist_ok=True)

        # Создаем папку для файла состояний, если нужно
        states_dir = os.path.dirname(states_file_abs)
        if states_dir:
            os.makedirs(states_dir, exist_ok=True)

        print(f"Папки созданы в: {project_root}")
        print(f"Папка данных: {data_folder_abs}")
        print(f"Файл состояний: {states_file_abs}")

    def setup_custom_logic(self):
        self.ui.stackedWidget.setCurrentIndex(0)
        """Настройка пользовательской логики"""
        self.update_file_list()

        # Подключение кнопок файлового менеджера
        self.ui.add_button.clicked.connect(self.add_file)
        self.ui.update_button.clicked.connect(self.update_file_list)
        self.ui.delete_button.clicked.connect(self.delete_file)
        self.ui.open_button.clicked.connect(self.open_file)
        self.ui.back_button.clicked.connect(self.go_back_to_main)

        # Подключение кнопок анализа
        self.ui.preprocessing_button.clicked.connect(self.open_preprocessing_window)
        self.ui.visualization_button.clicked.connect(self.open_visualization)
        self.ui.modeling_button.clicked.connect(self.open_modeling)

        # Начальное состояние кнопок анализа
        self.ui.visualization_button.setEnabled(False)
        self.ui.modeling_button.setEnabled(False)

        # Сбрасываем выделение в списке файлов
        self.ui.file_list.clearSelection()
        self.ui.add_button.setFocus()

        # Обновляем статистику файлов
        self.update_file_stats()

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

                # Проверяем и обновляем состояние файла
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

        # Сохраняем имя текущего файла
        self.current_filename = filename

        # Проверяем и обновляем состояние файла
        self.check_and_update_file_state(filename)

        # Обновляем состояние кнопок анализа для этого файла
        self.update_analysis_buttons_state()

        # Переходим на страницу анализа
        self.ui.stackedWidget.setCurrentIndex(1)
        self.ui.label_4.setText(
            f"<h1 style='color: #1e3a5f; margin: 15px; text-align: center;'>🔧 Анализ: {filename}</h1>")

        QMessageBox.information(self, "Успех",
                                f"Файл {filename} успешно загружен для анализа!")

    def check_and_update_file_state(self, filename):
        """Проверяем и обновляем состояние файла в CSV состояний"""
        try:
            # Создаем CSV файл состояний если его нет
            if not os.path.exists(self.states_file):
                df_states = pd.DataFrame(
                    columns=['name', 'preprocessing', 'visualization', 'modeling'])
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
                print(f"Добавлена новая запись для файла: {filename}")

        except Exception as e:
            print(f"Ошибка при обновлении состояния файла: {e}")

    def update_analysis_buttons_state(self):
        """Обновление состояния кнопок анализа"""
        if not self.current_filename:
            # Если файл не выбран, все кнопки кроме предобработки disabled
            self.set_buttons_state(True, False, False)
            return

        try:
            # Проверяем существует ли файл состояний
            if not os.path.exists(self.states_file):
                # Если файла нет, все кнопки кроме предобработки disabled
                self.set_buttons_state(True, False, False)
                return

            # Загружаем состояния из файла
            df_states = pd.read_csv(self.states_file)

            # Проверяем есть ли запись для текущего файла
            if self.current_filename in df_states['name'].values:
                # Получаем состояние для текущего файла
                file_state = df_states[df_states['name'] == self.current_filename].iloc[0]

                # Визуализация доступна только после предобработки
                visualization_enabled = bool(file_state['preprocessing'])
                # Моделирование доступно только после предобработки
                modeling_enabled = bool(file_state['preprocessing'])

                self.set_buttons_state(True, visualization_enabled, modeling_enabled)

                print(f"Состояние кнопок для {self.current_filename}:")
                print(f"  Предобработка: True")
                print(f"  Визуализация: {visualization_enabled}")
                print(f"  Моделирование: {modeling_enabled}")
            else:
                # Если записи нет, все кнопки кроме предобработки disabled
                self.set_buttons_state(True, False, False)

        except Exception as e:
            print(f"Ошибка при обновлении состояния кнопок: {e}")
            self.set_buttons_state(True, False, False)

    def set_buttons_state(self, preprocessing_enabled, visualization_enabled, modeling_enabled):
        """Устанавливает состояние кнопок анализа"""
        self.ui.preprocessing_button.setEnabled(preprocessing_enabled)
        self.ui.visualization_button.setEnabled(visualization_enabled)
        self.ui.modeling_button.setEnabled(modeling_enabled)

        # Обновляем подсказки для кнопок
        if not visualization_enabled:
            self.ui.visualization_button.setToolTip("Сначала завершите предобработку данных")
        else:
            self.ui.visualization_button.setToolTip("Открыть инструменты визуализации")

        if not modeling_enabled:
            self.ui.modeling_button.setToolTip("Сначала завершите предобработку данных")
        else:
            self.ui.modeling_button.setToolTip("Открыть инструменты моделирования")

    def update_file_state(self, step_name, completed=True):
        """Обновляет состояние конкретного шага для текущего файла"""
        if not self.current_filename:
            return

        try:
            # Загружаем текущие состояния
            if os.path.exists(self.states_file):
                df_states = pd.read_csv(self.states_file)
            else:
                df_states = pd.DataFrame(
                    columns=['name', 'preprocessing', 'visualization', 'modeling'])

            # Проверяем есть ли запись для этого файла
            if self.current_filename in df_states['name'].values:
                # Обновляем состояние
                df_states.loc[df_states['name'] == self.current_filename, step_name] = completed
                df_states.to_csv(self.states_file, index=False)

                print(
                    f"Обновлено состояние {step_name} для файла {self.current_filename}: {completed}")
            else:
                print(f"Файл {self.current_filename} не найден в состояниях")

            # После обновления состояния, обновляем кнопки
            self.update_analysis_buttons_state()

        except Exception as e:
            print(f"Ошибка при обновлении состояния файла: {e}")

    def open_preprocessing_window(self):
        """Открывает окно препроцессинга для текущего файла"""
        if not self.current_filename:
            QMessageBox.warning(self, "Ошибка", "Сначала выберите файл для анализа!")
            return

        # Проверяем, существует ли файл
        file_path = os.path.join(self.data_folder, self.current_filename)
        if not os.path.exists(file_path):
            QMessageBox.critical(self, "Ошибка", f"Файл {self.current_filename} не найден!")
            return

        # Скрываем главное окно
        self.hide()

        # Создаем окно препроцессинга с передачей текущего файла и родительского окна
        self.preprocessing_window = PreprocessingWindow(
            filename=self.current_filename,
            parent=self  # Важно: передаем self как родителя
        )

        # Подключаем сигнал закрытия окна препроцессинга
        self.preprocessing_window.closed.connect(self.on_preprocessing_closed)

        # Отображаем окно препроцессинга
        self.preprocessing_window.show()

        print(f"Открыто окно препроцессинга для файла: {self.current_filename}")

    def on_preprocessing_closed(self):
        """Обработчик закрытия окна препроцессинга"""
        print("Окно препроцессинга закрыто")

        # Отмечаем, что окно препроцессинга закрыто
        self.preprocessing_window = None

        # Показываем главное окно
        self.show()

        # Активируем главное окно
        self.activateWindow()
        self.raise_()

        # Обновляем состояние кнопок анализа
        self.update_analysis_buttons_state()

        # Обновляем список файлов (на случай, если данные были изменены)
        self.update_file_list()

    def open_visualization(self):
        """Открытие окна визуализации"""
        if not self.current_filename:
            QMessageBox.warning(self, "Ошибка", "Сначала выберите файл для анализа!")
            return

        # Проверяем, завершена ли предобработка
        if not self.is_preprocessing_completed():
            QMessageBox.warning(self, "Предупреждение",
                                "Сначала завершите предобработку данных!")
            return

        # TODO: Реализовать открытие окна визуализации
        QMessageBox.information(self, "Визуализация",
                                f"Открытие инструментов визуализации для файла: {self.current_filename}")

    def open_modeling(self):
        """Открытие окна моделирования"""
        if not self.current_filename:
            QMessageBox.warning(self, "Ошибка", "Сначала выберите файл для анализа!")
            return

        # Проверяем, завершена ли предобработка
        if not self.is_preprocessing_completed():
            QMessageBox.warning(self, "Предупреждение",
                                "Сначала завершите предобработку данных!")
            return

        # TODO: Реализовать открытие окна моделирования
        QMessageBox.information(self, "Моделирование",
                                f"Открытие инструментов моделирования для файла: {self.current_filename}")

    def is_preprocessing_completed(self):
        """Проверяет, завершена ли предобработка для текущего файла"""
        if not self.current_filename:
            return False

        try:
            if not os.path.exists(self.states_file):
                return False

            df_states = pd.read_csv(self.states_file)

            if self.current_filename in df_states['name'].values:
                file_state = df_states[df_states['name'] == self.current_filename].iloc[0]
                return bool(file_state['preprocessing'])
            else:
                return False

        except Exception as e:
            print(f"Ошибка при проверке завершения предобработки: {e}")
            return False

    def go_back_to_main(self):
        """Возврат на главную страницу"""
        self.ui.stackedWidget.setCurrentIndex(0)
        self.current_filename = None  # Сбрасываем текущий файл
        self.update_analysis_buttons_state()

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
        try:
            if os.path.exists(self.states_file):
                df_states = pd.read_csv(self.states_file)

                # Проверяем есть ли запись для этого файла
                if filename in df_states['name'].values:
                    # Удаляем строку с этим файлом
                    df_states = df_states[df_states['name'] != filename]
                    df_states.to_csv(self.states_file, index=False)

        except Exception as e:
            print(f"Ошибка при удалении состояния файла: {e}")

    def cleanup_orphaned_states(self):
        """Очищает состояния файлов, которых больше нет в папке данных"""
        try:
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

        except Exception as e:
            print(f"Ошибка при очистке orphaned состояний: {e}")

    def closeEvent(self, event):
        """Обработчик закрытия главного окна"""
        # Если открыто окно препроцессинга, закрываем его
        if self.preprocessing_window and self.preprocessing_window.isVisible():
            self.preprocessing_window.close()

        # Принимаем событие закрытия
        event.accept()
