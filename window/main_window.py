import os
import sys
import shutil
import pandas as pd
import json
from PyQt6.QtWidgets import (QApplication, QMainWindow, QFileDialog,
                             QMessageBox, QListWidgetItem)
from PyQt6.QtCore import Qt
from form.main_window_ui import Ui_Platform
from window.visualization_window import VisualizationWindow
from window.preprocessing_window import PreprocessingWindow
from window.modeling_window import ModelingWindow


class AppDataManager:
    """Менеджер для работы с данными приложения"""

    def __init__(self, app_data_dir: str = "data/app_data"):
        self.app_data_dir = app_data_dir
        os.makedirs(app_data_dir, exist_ok=True)
        self.states_file = os.path.join(app_data_dir, "file_states.csv")

    def init_states_file(self):
        """Инициализация файла состояний - только 3 колонки"""
        if not os.path.exists(self.states_file):
            df_states = pd.DataFrame(
                columns=['name', 'preprocessing', 'separator'])
            df_states.to_csv(self.states_file, index=False)
            print(f"Создан файл состояний: {self.states_file}")

    def add_file_state(self, filename, separator=','):
        """Добавляет запись о файле в состояния с разделителем по умолчанию ','"""
        try:
            self.init_states_file()

            df_states = pd.read_csv(self.states_file)

            if filename not in df_states['name'].values:
                new_row = pd.DataFrame({
                    'name': [filename],
                    'preprocessing': [False],
                    'separator': [separator]
                })
                df_states = pd.concat([df_states, new_row], ignore_index=True)
                df_states.to_csv(self.states_file, index=False)
                print(f"Добавлена запись для файла: {filename}, разделитель: '{separator}'")

        except Exception as e:
            print(f"Ошибка при добавлении состояния файла: {e}")

    def update_file_state(self, filename, preprocessing_completed=True, separator=None):
        """Обновляет состояние файла - только preprocessing и разделитель"""
        try:
            self.init_states_file()

            df_states = pd.read_csv(self.states_file)

            if filename not in df_states['name'].values:
                # Если файла нет в состояниях, добавляем его
                self.add_file_state(filename, separator or ',')
                df_states = pd.read_csv(self.states_file)  # Перечитываем

            # Обновляем только preprocessing
            if preprocessing_completed is not None:
                df_states.loc[
                    df_states['name'] == filename, 'preprocessing'] = preprocessing_completed

            # Обновляем разделитель если передан
            if separator is not None:
                df_states.loc[df_states['name'] == filename, 'separator'] = separator

            df_states.to_csv(self.states_file, index=False)

            print(
                f"Обновлено состояние для {filename}: preprocessing={preprocessing_completed}, separator='{separator}'")

        except Exception as e:
            print(f"Ошибка при обновлении состояния файла: {e}")

    def get_file_state(self, filename):
        """Получает состояние файла"""
        try:
            if not os.path.exists(self.states_file):
                return None

            df_states = pd.read_csv(self.states_file)

            if filename in df_states['name'].values:
                return df_states[df_states['name'] == filename].iloc[0]
            else:
                return None

        except Exception as e:
            print(f"Ошибка при получении состояния файла: {e}")
            return None

    def get_separator(self, filename):
        """Получает разделитель для файла"""
        state = self.get_file_state(filename)
        if state is not None and 'separator' in state:
            separator = state['separator']
            # Если separator NaN или пустой, возвращаем запятую по умолчанию
            if pd.isna(separator) or separator == '':
                return ','
            return separator
        return ','  # По умолчанию запятая

    def update_separator(self, filename, separator):
        """Обновляет только разделитель для файла"""
        self.update_file_state(filename, preprocessing_completed=None, separator=separator)

    def remove_file_state(self, filename):
        """Удаляет запись о файла из CSV состояний"""
        try:
            if os.path.exists(self.states_file):
                df_states = pd.read_csv(self.states_file)

                if filename in df_states['name'].values:
                    df_states = df_states[df_states['name'] != filename]
                    df_states.to_csv(self.states_file, index=False)
                    print(f"Удалено состояние файла: {filename}")

        except Exception as e:
            print(f"Ошибка при удалении состояния файла: {e}")

    def cleanup_dataset_files(self, dataset_name: str):
        """Очищает все конфигурационные файлы для датасета"""
        try:
            base_name = os.path.splitext(dataset_name)[0]
            files_to_remove = []

            # Ищем все файлы в app_data связанные с этим датасетом
            if os.path.exists(self.app_data_dir):
                for file in os.listdir(self.app_data_dir):
                    if file.startswith(base_name + '_'):
                        files_to_remove.append(os.path.join(self.app_data_dir, file))

            removed_count = 0
            for file_path in files_to_remove:
                try:
                    os.remove(file_path)
                    print(f"Удален файл конфигурации: {file_path}")
                    removed_count += 1
                except Exception as e:
                    print(f"Ошибка при удалении файла {file_path}: {e}")

            # Удаляем запись из файла состояний
            self.remove_file_state(dataset_name)

            return removed_count

        except Exception as e:
            print(f"Ошибка при очистке файлов датасета: {e}")
            return 0


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.data_folder = 'data/storage'
        self.app_data_manager = AppDataManager()
        self.current_filename = None
        self.preprocessing_window = None
        self.visualization_window = None
        self.modeling_window = None
        self._processing_matplotlib_close = False

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
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_file_dir)

        data_folder_abs = os.path.join(project_root, self.data_folder)
        os.makedirs(data_folder_abs, exist_ok=True)

        print(f"Папки созданы в: {project_root}")
        print(f"Папка данных: {data_folder_abs}")

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

            stats_text = f"Статистика: {len(files)} файлов (CSV: {csv_count}, JSON: {json_count})"
            self.ui.stats_label.setText(stats_text)

        except FileNotFoundError:
            self.ui.stats_label.setText("Папка не найдена")

    def update_file_list(self):
        """Обновление списка файлов"""
        self.ui.file_list.clear()
        try:
            files = os.listdir(self.data_folder)
            for file in files:
                if file.endswith(('.csv', '.json')):
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

                # Создаем начальное состояние файла с разделителем по умолчанию
                # Для CSV используем запятую, для JSON - None
                separator = ',' if filename.endswith('.csv') else ''
                self.app_data_manager.add_file_state(filename, separator)

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

        # Обновляем состояние кнопок анализа для этого файла
        self.update_analysis_buttons_state()

        # Переходим на страницу анализа
        self.ui.stackedWidget.setCurrentIndex(1)
        self.ui.label_4.setText(
            f"<h1 style='color: #1e3a5f; margin: 15px; text-align: center;'>Анализ: {filename}</h1>")

        QMessageBox.information(self, "Успех",
                                f"Файл {filename} успешно загружен для анализа!")

    def update_analysis_buttons_state(self):
        """Обновление состояния кнопок анализа"""
        if not self.current_filename:
            self.set_buttons_state(True, False, False)
            return

        try:
            file_state = self.app_data_manager.get_file_state(self.current_filename)

            if file_state is not None:
                # Визуализация и моделирование доступны только если preprocessing=True
                preprocessing_completed = bool(file_state['preprocessing'])

                self.set_buttons_state(True, preprocessing_completed, preprocessing_completed)

                print(f"Состояние кнопок для {self.current_filename}:")
                print(f"  Предобработка: True")
                print(f"  Визуализация: {preprocessing_completed}")
                print(f"  Моделирование: {preprocessing_completed}")
            else:
                self.set_buttons_state(True, False, False)

        except Exception as e:
            print(f"Ошибка при обновлении состояния кнопок: {e}")
            self.set_buttons_state(True, False, False)

    def set_buttons_state(self, preprocessing_enabled, visualization_enabled, modeling_enabled):
        """Устанавливает состояние кнопок анализа"""
        self.ui.preprocessing_button.setEnabled(preprocessing_enabled)
        self.ui.visualization_button.setEnabled(visualization_enabled)
        self.ui.modeling_button.setEnabled(modeling_enabled)

        if not visualization_enabled:
            self.ui.visualization_button.setToolTip("Сначала завершите предобработку данных")
        else:
            self.ui.visualization_button.setToolTip("Открыть инструменты визуализации")

        if not modeling_enabled:
            self.ui.modeling_button.setToolTip("Сначала завершите предобработку данных")
        else:
            self.ui.modeling_button.setToolTip("Открыть инструменты моделирования")

    def open_preprocessing_window(self):
        """Открывает окно препроцессинга для текущего файла"""
        if not self.current_filename:
            QMessageBox.warning(self, "Ошибка", "Сначала выберите файл для анализа!")
            return

        file_path = os.path.join(self.data_folder, self.current_filename)
        if not os.path.exists(file_path):
            QMessageBox.critical(self, "Ошибка", f"Файл {self.current_filename} не найден!")
            return

        # Скрываем главное окно
        self.hide()

        # Создаем окно препроцессинга
        self.preprocessing_window = PreprocessingWindow(
            filename=self.current_filename,
            parent=self
        )

        # Подключаем сигнал закрытия
        self.preprocessing_window.closed.connect(self.on_preprocessing_closed)

        # Отображаем окно препроцессинга
        self.preprocessing_window.show()

        print(f"Открыто окно препроцессинга для файла: {self.current_filename}")

    def on_preprocessing_closed(self):
        """Обработчик закрытия окна препроцессинга"""
        print("Окно препроцессинга закрыто")
        self.preprocessing_window = None
        self.show()
        self.activateWindow()
        self.raise_()
        self.update_analysis_buttons_state()
        self.update_file_list()

    def update_file_separated(self, separator):
        """Обновляет разделитель для текущего файла"""
        if self.current_filename and separator:
            self.app_data_manager.update_separator(self.current_filename, separator)
            print(f"Обновлен разделитель для {self.current_filename}: '{separator}'")

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

        # Проверяем, существует ли файл
        file_path = os.path.join(self.data_folder, self.current_filename)
        if not os.path.exists(file_path):
            QMessageBox.critical(self, "Ошибка", f"Файл {self.current_filename} не найден!")
            return

        print(f"Открываем визуализацию для {self.current_filename}")

        # Скрываем главное окно
        self.hide()

        # Создаем окно визуализации
        self.visualization_window = VisualizationWindow(
            filename=self.current_filename,
            parent=self
        )

        # Подключаем сигнал закрытия окна визуализации
        self.visualization_window.closed.connect(self.on_visualization_closed)

        # Отображаем окно визуализации
        self.visualization_window.show()

        print(f"Окно визуализации показано")

    def on_visualization_closed(self):
        """Обработчик закрытия окна визуализации"""
        print("Сигнал on_visualization_closed получен")

        # Удаляем ссылку на окно
        if self.visualization_window:
            print(f"Удаляем ссылку на окно визуализации")
            self.visualization_window = None

        # Показываем главное окно
        print("Показываем главное окно")
        self.show()
        self.raise_()
        self.activateWindow()

        # Обновляем список файлов
        self.update_file_list()
        print("Главное окно показано и активировано")

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

        # Проверяем, существует ли файл
        file_path = os.path.join(self.data_folder, self.current_filename)
        if not os.path.exists(file_path):
            QMessageBox.critical(self, "Ошибка", f"Файл {self.current_filename} не найден!")
            return

        # Скрываем главное окно
        self.hide()

        # Создаем окно моделирования
        self.modeling_window = ModelingWindow(
            filename=self.current_filename,
            parent=self
        )

        # Подключаем сигнал закрытия
        self.modeling_window.closed.connect(self.on_modeling_closed)

        # Отображаем окно моделирования
        self.modeling_window.show()

        print(f"Открыто окно моделирования для файла: {self.current_filename}")

    # Добавьте метод-обработчик:
    def on_modeling_closed(self):
        """Обработчик закрытия окна моделирования"""
        print("Окно моделирования закрыто")
        self.modeling_window = None
        self.show()
        self.activateWindow()
        self.raise_()

    def is_preprocessing_completed(self):
        """Проверяет, завершена ли предобработка для текущего файла"""
        if not self.current_filename:
            return False

        try:
            file_state = self.app_data_manager.get_file_state(self.current_filename)
            if file_state is not None:
                return bool(file_state['preprocessing'])
            else:
                return False

        except Exception as e:
            print(f"Ошибка при проверке завершения предобработки: {e}")
            return False

    def go_back_to_main(self):
        """Возврат на главную страницу"""
        self.ui.stackedWidget.setCurrentIndex(0)
        self.current_filename = None
        self.update_analysis_buttons_state()

    def delete_file(self):
        """Удаление выбранного файла с очисткой конфигураций"""
        current_item = self.ui.file_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Предупреждение", "Выберите файл для удаления!")
            return

        filename = current_item.text().split(" ", 1)[1]
        file_path = os.path.join(self.data_folder, filename)

        reply = QMessageBox.question(
            self,
            "Подтверждение удаления",
            f"Вы уверены, что хотите удалить датасет '{filename}'?\n"
            f"Все связанные конфигурации (графики, настройки) также будут удалены.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                # Очищаем конфигурационные файлы
                removed_configs = self.app_data_manager.cleanup_dataset_files(filename)

                # Удаляем основной файл
                os.remove(file_path)

                # Обновляем интерфейс
                self.update_file_list()

                message = f"Датасет '{filename}' удален!"
                if removed_configs > 0:
                    message += f"\nУдалено {removed_configs} файлов конфигурации."

                QMessageBox.information(self, "Успех", message)

            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось удалить файл: {str(e)}")

    def cleanup_orphaned_states(self):
        """Очищает состояния файлов, которых больше нет в папке данных"""
        try:
            if not os.path.exists(self.app_data_manager.states_file):
                return

            df_states = pd.read_csv(self.app_data_manager.states_file)
            existing_files = set(os.listdir(self.data_folder))

            # Находим файлы в состояниях, которых нет в папке
            orphaned_files = df_states[~df_states['name'].isin(existing_files)]

            if not orphaned_files.empty:
                # Удаляем orphaned записи и их конфигурации
                for _, row in orphaned_files.iterrows():
                    filename = row['name']
                    self.app_data_manager.cleanup_dataset_files(filename)
                    print(f"Очищены конфигурации для отсутствующего файла: {filename}")

                # Обновляем файл состояний
                df_states = df_states[df_states['name'].isin(existing_files)]
                df_states.to_csv(self.app_data_manager.states_file, index=False)

        except Exception as e:
            print(f"Ошибка при очистке orphaned состояний: {e}")

    def closeEvent(self, event):
        """Обработчик закрытия главного окна"""
        print(
            f"closeEvent главного окна: _processing_matplotlib_close={self._processing_matplotlib_close}")

        # Если это закрытие из-за matplotlib окон - игнорируем
        if self._processing_matplotlib_close:
            print("Игнорируем closeEvent главного окна (вызвано matplotlib)")
            event.ignore()
            return

        # Нормальное закрытие главного окна
        print("Нормальное закрытие главного окна")

        # Закрываем дочерние окна если они открыты
        if self.preprocessing_window and self.preprocessing_window.isVisible():
            print("Закрываем окно препроцессинга")
            self.preprocessing_window.close()

        if self.visualization_window and self.visualization_window.isVisible():
            print("Закрываем окно визуализации")
            self.visualization_window.close()

        if self.modeling_window and self.modeling_window.isVisible():
            print("Закрываем окно моделирования")
            self.modeling_window.close()

        event.accept()
        QApplication.instance().quit()

    # НОВЫЕ МЕТОДЫ ДЛЯ ОБНОВЛЕНИЯ СОСТОЯНИЯ ФАЙЛА
    def update_file_state(self, state_type, completed=True):
        """Обновляет состояние файла для родительского окна"""
        if not self.current_filename:
            return

        try:
            if state_type == 'preprocessing':
                # Обновляем состояние препроцессинга в файле состояний
                self.app_data_manager.update_file_state(
                    self.current_filename,
                    preprocessing_completed=completed
                )
                print(f"Обновлено состояние предобработки для {self.current_filename}: {completed}")

            # Обновляем состояние кнопок
            self.update_analysis_buttons_state()

        except Exception as e:
            print(f"Ошибка при обновлении состояния файла: {e}")

    def update_file_separator(self, separator):
        """Обновляет разделитель для текущего файла"""
        if self.current_filename and separator:
            self.app_data_manager.update_separator(self.current_filename, separator)
            print(f"Обновлен разделитель для {self.current_filename}: '{separator}'")