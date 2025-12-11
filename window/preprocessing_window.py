import os
import pandas as pd
import numpy as np
import tempfile
import matplotlib

matplotlib.use('Agg')  # Используем бэкенд без GUI
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (QMainWindow, QMessageBox, QWidget, QTableWidgetItem, QHeaderView)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap
from form.preproccesing_window_ui import Ui_Preprocessing


class PreprocessingWindow(QMainWindow):
    closed = pyqtSignal()  # Сигнал при закрытии окна

    def __init__(self, filename, parent=None):
        super().__init__(parent)
        self.filename = filename
        self.parent_window = parent  # Сохраняем ссылку на родительское окно (MainWindow)
        self.data = None
        self.file_path = os.path.join("data/storage", self.filename)

        # Список для хранения временных файлов (графиков)
        self.temp_files = []

        # Флаг, указывающий были ли изменения
        self.data_changed = False

        # Инициализация UI
        self.ui = Ui_Preprocessing()
        self.ui.setupUi(self)

        # Настройка окна
        self.setWindowTitle(f"DataLite - Предобработка данных: {self.filename}")

        # Загрузка данных
        if not self.load_data():
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить файл: {self.filename}")
            self.close_window()
            return

        # Настройка UI
        self.setup_ui()

        # Подключение сигналов
        self.connect_signals()

        # Начальное состояние
        self.reset_ui_state()

    def reset_ui_state(self):
        """Сброс состояния UI к начальному"""
        # Показываем только секцию выбора на страницах пропусков и замены
        self.ui.pass_selection_frame.setVisible(True)
        self.ui.pass_numeric_frame.setVisible(False)
        self.ui.pass_categorical_frame.setVisible(False)

        self.ui.replace_selection_frame.setVisible(True)
        self.ui.replace_numeric_frame.setVisible(False)
        self.ui.replace_categorical_frame.setVisible(False)

    def setup_ui(self):
        """Настройка пользовательского интерфейса"""
        # Обновляем заголовки с именем файла
        self.ui.content_text_edit.setReadOnly(True)
        self.update_page_titles()

        # Настраиваем кнопки навигации
        self.setup_navigation()

        # Инициализируем страницы
        self.init_view_page()

    def update_page_titles(self):
        """Обновление заголовков страниц с именем файла"""
        # Страница просмотра файла
        self.ui.label_5.setText(
            f"<h1 style='color: #1e3a5f; margin: 20px; text-align: center; font-size: 26px;'> Просмотр файла: {self.filename} </h1>")

        # Страница анализа столбцов
        self.ui.label_8.setText(
            f"<h1 style='color: #1e3a5f; margin: 20px; text-align: center; font-size: 26px;'> Анализ столбцов: {self.filename} </h1>")

        # Страница обработки пропусков
        self.ui.pass_title_label.setText(
            f"<h1 style='color: #1e3a5f; margin: 10px; text-align: center; font-size: 26px;'> Обработка пропусков: {self.filename} </h1>")

        # Страница замены данных
        self.ui.replace_title_label.setText(
            f"<h1 style='color: #1e3a5f; margin: 10px; text-align: center; font-size: 26px;'> Замена данных: {self.filename} </h1>")

        # Страница обработки дубликатов
        self.ui.title_label.setText(
            f"<h1 style='color: #1e3a5f; margin: 10px; text-align: center; font-size: 26px;'> Обработка дубликатов: {self.filename} </h1>")

        # Страница удаления пропусков
        self.ui.title_label_missing.setText(
            f"<h1 style='color: #1e3a5f; margin: 10px; text-align: center; font-size: 26px;'> Удаление пропусков: {self.filename} </h1>")

        self.ui.column_ops_title_label.setText(
            f"<h1 style='color: #1e3a5f; margin: 10px; text-align: center; font-size: 26px;'> Управление столбцами: {self.filename} </h1>")

    def setup_navigation(self):
        """Настройка навигации между страницами"""
        # Устанавливаем начальную страницу
        self.ui.stackedWidget.setCurrentIndex(0)  # Страница просмотра

        # Блокируем кнопку "Назад" на первой странице
        self.ui.back_button.setEnabled(False)

        # Активируем кнопку "Продолжить"
        self.ui.next_button.setEnabled(True)

        # Блокируем кнопку "Завершить" пока не пройдем все шаги
        self.ui.compete_button.setEnabled(False)

    def connect_signals(self):
        """Подключение сигналов кнопок"""
        # Навигация
        self.ui.go_back.clicked.connect(self.on_close_button_clicked)
        self.ui.back_button.clicked.connect(self.go_back)
        self.ui.next_button.clicked.connect(self.go_next)
        self.ui.compete_button.clicked.connect(self.complete_preprocessing)

        # Кнопки разделителей для CSV
        self.ui.comma_button.clicked.connect(lambda: self.select_delimiter(","))
        self.ui.semicolon_button.clicked.connect(lambda: self.select_delimiter(";"))
        self.ui.tab_button.clicked.connect(lambda: self.select_delimiter("\t"))
        self.ui.apply_delimiter_button.clicked.connect(self.apply_delimiter)

        # Кнопки анализа столбцов
        self.ui.auto_detect_button.clicked.connect(self.auto_detect_data_type)
        self.ui.apply_dtype_button.clicked.connect(self.apply_data_type)

        # Кнопки обработки пропусков
        self.ui.pass_numeric_button.clicked.connect(self.open_numeric_missing)
        self.ui.pass_categorical_button.clicked.connect(self.open_categorical_missing)
        self.ui.pass_back_numeric_button.clicked.connect(self.go_back_to_missing_selection)
        self.ui.pass_back_category_button.clicked.connect(self.go_back_to_missing_selection)
        self.ui.pass_apply_numeric_button.clicked.connect(self.apply_numeric_missing)
        self.ui.pass_apply_category_button.clicked.connect(self.apply_categorical_missing)

        # Кнопки замены данных
        self.ui.replace_numeric_button.clicked.connect(self.open_numeric_replace)
        self.ui.replace_categorical_button.clicked.connect(self.open_categorical_replace)
        self.ui.replace_numeric_back_button.clicked.connect(self.go_back_to_replace_selection)
        self.ui.replace_categorical_back_button.clicked.connect(self.go_back_to_replace_selection)

        # Подключение обработчиков выбора столбцов
        self.ui.pass_column_combo.currentTextChanged.connect(self.update_missing_stats)
        self.ui.replace_column_listwidget.itemSelectionChanged.connect(
            self.on_replace_column_selected)

        # Подключение методов обработки для страницы замены
        self.ui.replace_numeric_apply_button.clicked.connect(self.apply_numeric_replace)
        self.ui.replace_categorical_apply_button.clicked.connect(self.apply_categorical_replace)

        # Подключение обработчиков для страницы замены
        self.ui.replace_numeric_radio_greater.toggled.connect(self.update_replace_numeric_preview)
        self.ui.replace_numeric_radio_less.toggled.connect(self.update_replace_numeric_preview)
        self.ui.replace_numeric_radio_greater_equal.toggled.connect(
            self.update_replace_numeric_preview)
        self.ui.replace_numeric_radio_less_equal.toggled.connect(
            self.update_replace_numeric_preview)
        self.ui.replace_numeric_threshold_edit.textChanged.connect(
            self.update_replace_numeric_preview)
        self.ui.replace_numeric_radio_multiply.toggled.connect(self.update_replace_numeric_preview)
        self.ui.replace_numeric_radio_delete.toggled.connect(self.update_replace_numeric_preview)
        self.ui.replace_numeric_multiply_edit.textChanged.connect(
            self.update_replace_numeric_preview)

        # Подключение обработчиков для страницы категориальной замены
        self.ui.replace_categorical_search_edit.textChanged.connect(self.filter_categorical_values)
        self.ui.replace_categorical_values_list.itemSelectionChanged.connect(
            self.update_categorical_preview)
        self.ui.replace_categorical_replace_with_edit.textChanged.connect(
            self.update_categorical_preview)

        # Подключение обработчиков для страницы дубликатов
        self.ui.remove_duplicates_btn.clicked.connect(self.remove_duplicates)
        self.ui.keep_duplicates_btn.clicked.connect(self.keep_duplicates)

        # Подключение обработчиков для страницы удаления пропусков
        self.ui.column_combo_missing.currentTextChanged.connect(self.update_missing_info)
        self.ui.clear_column_btn.clicked.connect(self.clear_column_missing)
        self.ui.clear_all_btn.clicked.connect(self.clear_all_missing)

        # Новые сигналы для управления столбцами
        self.ui.remove_column_button.clicked.connect(self.remove_selected_columns)
        self.ui.group_columns_listwidget.itemSelectionChanged.connect(self.update_agg_column_list)
        self.ui.agg_column_listwidget.itemSelectionChanged.connect(self.update_agg_preview)
        self.ui.group_columns_listwidget.itemSelectionChanged.connect(self.update_agg_preview)
        self.ui.agg_sum_radio.toggled.connect(self.update_agg_preview)
        self.ui.agg_mean_radio.toggled.connect(self.update_agg_preview)
        self.ui.agg_median_radio.toggled.connect(self.update_agg_preview)
        self.ui.agg_count_radio.toggled.connect(self.update_agg_preview)
        self.ui.agg_min_radio.toggled.connect(self.update_agg_preview)
        self.ui.agg_max_radio.toggled.connect(self.update_agg_preview)
        self.ui.agg_value_counts_radio.toggled.connect(self.update_agg_preview)
        self.ui.new_agg_column_name_edit.textChanged.connect(self.update_agg_preview)
        self.ui.create_agg_column_button.clicked.connect(self.create_agg_column)

    def load_column_operations_page(self):
        """Загрузка данных для страницы управления столбцами"""
        try:
            if self.data is None:
                return

            # Заполняем список столбцов для удаления
            self.ui.remove_column_listwidget.clear()
            self.ui.remove_column_listwidget.addItems(self.data.columns)

            # Заполняем списки для группировки и агрегации
            self.ui.group_columns_listwidget.clear()
            self.ui.agg_column_listwidget.clear()

            # Добавляем только числовые столбцы для агрегации
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
            all_columns = self.data.columns.tolist()

            self.ui.group_columns_listwidget.addItems(all_columns)
            self.ui.agg_column_listwidget.addItems(numeric_columns)

            # Сбрасываем поля
            self.ui.new_agg_column_name_edit.clear()
            self.ui.agg_preview_table.setRowCount(0)
            self.ui.agg_preview_table.setColumnCount(0)
            self.ui.agg_preview_info_label.setText("Выберите настройки для предпросмотра")

            # Устанавливаем стандартные значения
            self.ui.agg_sum_radio.setChecked(True)

            # Обновляем кнопку удаления
            self.ui.remove_column_button.setEnabled(False)
            self.ui.remove_column_listwidget.itemSelectionChanged.connect(
                self.update_remove_button_state)

            # Обновляем кнопку создания агрегатного столбца
            self.ui.create_agg_column_button.setEnabled(False)

        except Exception as e:
            print(f"Ошибка при загрузке страницы управления столбцами: {str(e)}")

    def update_remove_button_state(self):
        """Обновление состояния кнопки удаления столбцов"""
        selected_items = self.ui.remove_column_listwidget.selectedItems()
        has_selection = len(selected_items) > 0
        self.ui.remove_column_button.setEnabled(has_selection)

        # Нельзя удалить все столбцы
        if has_selection and len(selected_items) == len(self.data.columns):
            self.ui.remove_column_button.setEnabled(False)
            QMessageBox.warning(self, "Предупреждение",
                                "Нельзя удалить все столбцы из набора данных!")

    def remove_selected_columns(self):
        """Удаление выбранных столбцов"""
        try:
            selected_items = self.ui.remove_column_listwidget.selectedItems()
            if not selected_items:
                QMessageBox.warning(self, "Предупреждение", "Выберите столбцы для удаления!")
                return

            columns_to_remove = [item.text() for item in selected_items]

            # Проверяем, не пытаемся ли удалить все столбцы
            if len(columns_to_remove) == len(self.data.columns):
                QMessageBox.warning(self, "Ошибка",
                                    "Нельзя удалить все столбцы из набора данных!")
                return

            # Подтверждение удаления
            column_list = "\n".join([f"• {col}" for col in columns_to_remove])
            reply = QMessageBox.question(
                self, "Подтверждение удаления",
                f"Вы действительно хотите удалить следующие столбцы?\n\n"
                f"{column_list}\n\n"
                f"Всего: {len(columns_to_remove)} столбцов\n"
                f"После удаления останется {len(self.data.columns) - len(columns_to_remove)} столбцов.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.No:
                return

            # Сохраняем старые названия для отчета
            old_columns = list(self.data.columns)

            # Удаляем столбцы
            self.data = self.data.drop(columns=columns_to_remove)

            # Помечаем, что данные были изменены
            self.data_changed = True

            # Обновляем UI
            self.load_column_operations_page()

            # Показываем отчет
            QMessageBox.information(
                self, "Успех",
                f"Столбцы успешно удалены!\n\n"
                f"Удалено столбцов: {len(columns_to_remove)}\n"
                f"Осталось столбцов: {len(self.data.columns)}\n\n"
                f"Удаленные столбцы:\n{column_list}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при удалении столбцов: {str(e)}")

    def update_agg_column_list(self):
        """Обновление списка столбцов для агрегации после выбора группировки"""
        try:
            selected_group_items = self.ui.group_columns_listwidget.selectedItems()

            if not selected_group_items:
                # Если нет выбранных столбцов для группировки, показываем все числовые столбцы
                numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
                self.ui.agg_column_listwidget.clear()
                self.ui.agg_column_listwidget.addItems(numeric_columns)
                return

            # Получаем выбранные столбцы группировки
            group_columns = [item.text() for item in selected_group_items]

            # Определяем, какие столбцы можно использовать для агрегации
            # Нельзя агрегировать столбцы группировки сами по себе
            available_columns = []
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()

            for col in numeric_columns:
                if col not in group_columns:
                    available_columns.append(col)

            # Обновляем список
            self.ui.agg_column_listwidget.clear()
            self.ui.agg_column_listwidget.addItems(available_columns)

            # Если есть доступные столбцы, выбираем первый
            if available_columns:
                self.ui.agg_column_listwidget.setCurrentRow(0)

        except Exception as e:
            print(f"Ошибка при обновлении списка агрегации: {str(e)}")

    def update_agg_preview(self):
        """Обновление предпросмотра агрегатного столбца"""
        try:
            # Проверяем все необходимые условия
            column_name = self.ui.new_agg_column_name_edit.text().strip()
            selected_group_items = self.ui.group_columns_listwidget.selectedItems()
            selected_agg_items = self.ui.agg_column_listwidget.selectedItems()

            if not column_name:
                self.ui.agg_preview_info_label.setText("Введите имя нового столбца")
                self.clear_preview_table(self.ui.agg_preview_table)
                self.ui.create_agg_column_button.setEnabled(False)
                return

            if not selected_group_items:
                self.ui.agg_preview_info_label.setText("Выберите столбцы для группировки")
                self.clear_preview_table(self.ui.agg_preview_table)
                self.ui.create_agg_column_button.setEnabled(False)
                return

            if not selected_agg_items:
                self.ui.agg_preview_info_label.setText("Выберите столбец для агрегации")
                self.clear_preview_table(self.ui.agg_preview_table)
                self.ui.create_agg_column_button.setEnabled(False)
                return

            # Получаем выбранные столбцы
            group_columns = [item.text() for item in selected_group_items]
            agg_column = selected_agg_items[0].text()

            # Определяем выбранную операцию
            if self.ui.agg_sum_radio.isChecked():
                operation = "sum"
                operation_name = "Сумма"
            elif self.ui.agg_mean_radio.isChecked():
                operation = "mean"
                operation_name = "Среднее"
            elif self.ui.agg_median_radio.isChecked():
                operation = "median"
                operation_name = "Медиана"
            elif self.ui.agg_count_radio.isChecked():
                operation = "count"
                operation_name = "Количество"
            elif self.ui.agg_min_radio.isChecked():
                operation = "min"
                operation_name = "Минимум"
            elif self.ui.agg_max_radio.isChecked():
                operation = "max"
                operation_name = "Максимум"
            elif self.ui.agg_value_counts_radio.isChecked():
                operation = "value_counts"
                operation_name = "Частота значений"
            else:
                self.ui.agg_preview_info_label.setText("Выберите операцию агрегации")
                self.clear_preview_table(self.ui.agg_preview_table)
                self.ui.create_agg_column_button.setEnabled(False)
                return

            # Выполняем агрегацию для предпросмотра
            try:
                if operation == "value_counts":
                    # Для value_counts создаем pivot таблицу
                    grouped = self.data.groupby(group_columns)[agg_column].value_counts().unstack(
                        fill_value=0)
                    preview_data = grouped.head(10)  # Показываем первые 10 строк
                    column_count = len(preview_data.columns)

                    # Обновляем таблицу
                    self.ui.agg_preview_table.setRowCount(len(preview_data))
                    self.ui.agg_preview_table.setColumnCount(len(group_columns) + column_count)

                    # Устанавливаем заголовки
                    headers = group_columns + [f"{agg_column}_{col}" for col in
                                               preview_data.columns]
                    self.ui.agg_preview_table.setHorizontalHeaderLabels(headers)

                    # Заполняем таблицу
                    for i, (index, row) in enumerate(preview_data.iterrows()):
                        # Индекс может быть кортежем для нескольких столбцов группировки
                        if isinstance(index, tuple):
                            for j, value in enumerate(index):
                                self.ui.agg_preview_table.setItem(i, j,
                                                                  QTableWidgetItem(str(value)))
                        else:
                            self.ui.agg_preview_table.setItem(i, 0, QTableWidgetItem(str(index)))

                        # Заполняем значения агрегации
                        for j, value in enumerate(row):
                            item = QTableWidgetItem(str(value))
                            item.setBackground(Qt.GlobalColor.lightGray)
                            self.ui.agg_preview_table.setItem(i, len(group_columns) + j, item)

                    info_text = (f"Будет создано {column_count} новых столбцов\n"
                                 f"Операция: {operation_name}")

                else:
                    # Для обычных операций агрегации
                    grouped = self.data.groupby(group_columns)[agg_column].agg(operation)
                    preview_data = grouped.reset_index().head(10)  # Показываем первые 10 строк

                    # Обновляем таблицу
                    self.ui.agg_preview_table.setRowCount(len(preview_data))
                    self.ui.agg_preview_table.setColumnCount(len(preview_data.columns))

                    # Устанавливаем заголовки
                    headers = list(preview_data.columns)
                    headers[-1] = column_name  # Последний столбец - результат агрегации
                    self.ui.agg_preview_table.setHorizontalHeaderLabels(headers)

                    # Заполняем таблицу
                    for i, (idx, row) in enumerate(preview_data.iterrows()):
                        for j, col in enumerate(preview_data.columns):
                            value = row[col]
                            item = QTableWidgetItem(str(value))

                            # Подсвечиваем столбец с результатом агрегации
                            if j == len(preview_data.columns) - 1:
                                item.setBackground(Qt.GlobalColor.lightGray)

                            self.ui.agg_preview_table.setItem(i, j, item)

                    info_text = f"Будет создан столбец '{column_name}'\nОперация: {operation_name}"

                # Настраиваем заголовки таблицы
                header = self.ui.agg_preview_table.horizontalHeader()
                header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

                # Обновляем информацию
                self.ui.agg_preview_info_label.setText(info_text)
                self.ui.create_agg_column_button.setEnabled(True)

            except Exception as e:
                self.ui.agg_preview_info_label.setText(f"Ошибка при агрегации: {str(e)}")
                self.clear_preview_table(self.ui.agg_preview_table)
                self.ui.create_agg_column_button.setEnabled(False)

        except Exception as e:
            print(f"Ошибка при обновлении предпросмотра агрегации: {str(e)}")
            self.ui.create_agg_column_button.setEnabled(False)

    def create_agg_column(self):
        """Создание нового столбца через агрегацию"""
        try:
            # Получаем параметры
            new_column_name = self.ui.new_agg_column_name_edit.text().strip()
            selected_group_items = self.ui.group_columns_listwidget.selectedItems()
            selected_agg_items = self.ui.agg_column_listwidget.selectedItems()

            # Проверяем обязательные поля
            if not new_column_name:
                QMessageBox.warning(self, "Предупреждение", "Введите имя нового столбца!")
                return

            if not selected_group_items:
                QMessageBox.warning(self, "Предупреждение", "Выберите столбцы для группировки!")
                return

            if not selected_agg_items:
                QMessageBox.warning(self, "Предупреждение", "Выберите столбец для агрегации!")
                return

            # Проверяем, существует ли уже столбец с таким именем
            if new_column_name in self.data.columns:
                reply = QMessageBox.question(
                    self, "Подтверждение",
                    f"Столбец с именем '{new_column_name}' уже существует.\n"
                    f"Заменить существующий столбец?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    return

            # Получаем выбранные столбцы
            group_columns = [item.text() for item in selected_group_items]
            agg_column = selected_agg_items[0].text()

            # Определяем операцию
            if self.ui.agg_sum_radio.isChecked():
                operation = "sum"
                operation_name = "сумма"
            elif self.ui.agg_mean_radio.isChecked():
                operation = "mean"
                operation_name = "среднее"
            elif self.ui.agg_median_radio.isChecked():
                operation = "median"
                operation_name = "медиана"
            elif self.ui.agg_count_radio.isChecked():
                operation = "count"
                operation_name = "количество"
            elif self.ui.agg_min_radio.isChecked():
                operation = "min"
                operation_name = "минимум"
            elif self.ui.agg_max_radio.isChecked():
                operation = "max"
                operation_name = "максимум"
            elif self.ui.agg_value_counts_radio.isChecked():
                return self.create_value_counts_columns(new_column_name, group_columns, agg_column)
            else:
                QMessageBox.warning(self, "Предупреждение", "Выберите операцию агрегации!")
                return

            try:
                # Выполняем группировку и агрегацию
                grouped = self.data.groupby(group_columns)[agg_column].transform(operation)

                # Создаем новый столбец
                self.data[new_column_name] = grouped

                # Помечаем, что данные были изменены
                self.data_changed = True

                # Формируем отчет
                unique_groups = self.data[group_columns].drop_duplicates().shape[0]

                QMessageBox.information(
                    self, "Успех",
                    f"Новый столбец успешно создан!\n\n"
                    f"Имя столбца: {new_column_name}\n"
                    f"Операция: {operation_name}\n"
                    f"Столбец агрегации: {agg_column}\n"
                    f"Столбцы группировки: {', '.join(group_columns)}\n"
                    f"Уникальных групп: {unique_groups}\n"
                    f"Добавлено значений: {len(self.data)}"
                )

                # Обновляем UI
                self.load_column_operations_page()

            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка при создании столбца: {str(e)}")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при создании столбца: {str(e)}")

    def create_value_counts_columns(self, base_name, group_columns, agg_column):
        """Создание нескольких столбцов для value_counts"""
        try:
            # Получаем уникальные значения в агрегируемом столбце
            unique_values = self.data[agg_column].dropna().unique()

            if len(unique_values) == 0:
                QMessageBox.warning(self, "Предупреждение",
                                    f"Столбец '{agg_column}' не содержит данных!")
                return

            if len(unique_values) > 20:
                reply = QMessageBox.question(
                    self, "Подтверждение",
                    f"Столбец '{agg_column}' содержит {len(unique_values)} уникальных значений.\n"
                    f"Это создаст {len(unique_values)} новых столбцов.\n"
                    f"Продолжить?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    return

            # Создаем pivot таблицу
            pivot_table = pd.crosstab(
                index=[self.data[col] for col in group_columns],
                columns=self.data[agg_column],
                rownames=group_columns,
                colnames=[agg_column]
            ).reset_index()

            # Переименовываем столбцы
            new_columns = []
            for col in pivot_table.columns:
                if col in group_columns:
                    new_columns.append(col)
                else:
                    new_col_name = f"{base_name}_{col}"
                    new_columns.append(new_col_name)

            pivot_table.columns = new_columns

            # Объединяем с исходными данными
            self.data = self.data.merge(pivot_table, on=group_columns, how='left')

            # Помечаем, что данные были изменены
            self.data_changed = True

            # Формируем отчет
            created_columns = [col for col in new_columns if col not in group_columns]

            QMessageBox.information(
                self, "Успех",
                f"Создано {len(created_columns)} новых столбцов!\n\n"
                f"Базовое имя: {base_name}\n"
                f"Столбец агрегации: {agg_column}\n"
                f"Столбцы группировки: {', '.join(group_columns)}\n"
                f"Созданные столбцы:\n" + "\n".join([f"• {col}" for col in created_columns[:10]])
                + (f"\n... и еще {len(created_columns) - 10} столбцов"
                   if len(created_columns) > 10 else "")
            )

            # Обновляем UI
            self.load_column_operations_page()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка",
                                 f"Ошибка при создании столбцов value_counts: {str(e)}")

    def clear_preview_table(self, table_widget):
        """Очистка таблицы предпросмотра"""
        table_widget.setRowCount(0)
        table_widget.setColumnCount(0)

    def update_navigation_buttons(self):
        """Обновление состояния кнопок навигации"""
        current_index = self.ui.stackedWidget.currentIndex()
        max_index = self.ui.stackedWidget.count() - 1

        # Кнопка "Назад"
        self.ui.back_button.setEnabled(current_index > 0)

        # Кнопка "Продолжить"
        self.ui.next_button.setEnabled(current_index < max_index)

        # Кнопка "Завершить" - всегда активна на последней странице
        self.ui.compete_button.setEnabled(current_index == max_index)

    def load_data(self):
        """Загрузка данных из файла"""
        try:
            if not os.path.exists(self.file_path):
                QMessageBox.critical(self, "Ошибка", f"Файл '{self.filename}' не найден!")
                return False

            # Определяем тип файла
            if self.filename.endswith('.csv'):
                # Пробуем разные разделители для CSV
                delimiters = [',', ';', '\t', '|']
                for delim in delimiters:
                    try:
                        self.data = pd.read_csv(self.file_path, delimiter=delim, encoding='utf-8')
                        if len(self.data.columns) > 1:  # Успешно разделились на несколько колонок
                            self.current_delimiter = delim
                            self.ui.delimiter_edit.setText(delim)
                            self.select_delimiter_ui(delim)
                            break
                    except:
                        continue

                # Если не удалось прочитать с разделителями, пробуем без указания
                if self.data is None or len(self.data.columns) <= 1:
                    try:
                        self.data = pd.read_csv(self.file_path, encoding='utf-8')
                        if ',' in self.data.columns[0]:
                            self.current_delimiter = ','
                        elif ';' in self.data.columns[0]:
                            self.current_delimiter = ';'
                        else:
                            self.current_delimiter = ','
                        self.ui.delimiter_edit.setText(self.current_delimiter)
                    except Exception as e:
                        QMessageBox.critical(self, "Ошибка",
                                             f"Не удалось прочитать CSV файл: {str(e)}")
                        return False

            elif self.filename.endswith('.json'):
                try:
                    self.data = pd.read_json(self.file_path, orient='records', encoding='utf-8')
                    self.current_delimiter = None
                except Exception as e:
                    QMessageBox.critical(self, "Ошибка",
                                         f"Не удалось прочитать JSON файл: {str(e)}")
                    return False
            else:
                QMessageBox.critical(self, "Ошибка",
                                     f"Неподдерживаемый формат файла: {self.filename}")
                return False

            # Проверяем, что данные загружены
            if self.data is None or self.data.empty:
                QMessageBox.warning(self, "Предупреждение",
                                    f"Файл '{self.filename}' пуст или содержит некорректные данные!")
                return False

            # Создаем копию данных для возможного отката
            self.original_data = self.data.copy()

            return True

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при загрузке файла: {str(e)}")
            return False

    def init_view_page(self):
        """Инициализация страницы просмотра файла"""
        if self.data is None:
            return

        # Отображаем информацию о файле
        self.display_file_info()

        # Отображаем содержимое файла
        self.display_file_content()

        # Блокируем разделители для JSON файлов
        if self.filename.endswith('.json'):
            self.disable_delimiter_controls()

    def disable_delimiter_controls(self):
        """Блокирует элементы управления разделителем для JSON файлов"""
        self.ui.delimiter_edit.setEnabled(False)
        self.ui.comma_button.setEnabled(False)
        self.ui.semicolon_button.setEnabled(False)
        self.ui.tab_button.setEnabled(False)
        self.ui.apply_delimiter_button.setEnabled(False)

        # Добавляем подсказку
        self.ui.delimiter_edit.setPlaceholderText("Недоступно для JSON файлов")

        # Скрываем или меняем текст метки
        self.ui.label_7.setText("Разделитель: (недоступно для JSON)")

    def enable_delimiter_controls(self):
        """Разблокирует элементы управления разделителем для CSV файлов"""
        self.ui.delimiter_edit.setEnabled(True)
        self.ui.comma_button.setEnabled(True)
        self.ui.semicolon_button.setEnabled(True)
        self.ui.tab_button.setEnabled(True)
        self.ui.apply_delimiter_button.setEnabled(True)

        # Восстанавливаем стандартный placeholder
        self.ui.delimiter_edit.setPlaceholderText(",")
        self.ui.label_7.setText("Разделитель:")

    def display_file_info(self):
        """Отображение информации о файле"""
        try:
            file_size = os.path.getsize(self.file_path)
            file_size_kb = file_size / 1024

            info_text = f"""
             Файл: {self.filename}
             Размер: {file_size_kb:.2f} KB
             Строк: {len(self.data):,}
             Столбцов: {len(self.data.columns)}
             Пропусков: {self.data.isnull().sum().sum():,}
            """

            self.ui.file_info_label.setText(info_text)

        except Exception as e:
            self.ui.file_info_label.setText(f"Ошибка при получении информации: {str(e)}")

    def display_file_content(self):
        """Отображение содержимого файла"""
        try:
            if self.data is None:
                self.ui.content_text_edit.setText("Данные не загружены")
                return

            # Ограничиваем количество строк для отображения
            display_rows = min(100, len(self.data))

            # Если это CSV с разделителем
            if self.filename.endswith('.csv') and hasattr(self, 'current_delimiter'):
                content = ""
                # Добавляем заголовки
                content += self.current_delimiter.join(self.data.columns.astype(str)) + "\n"

                # Добавляем данные
                for i in range(display_rows):
                    row = self.current_delimiter.join(self.data.iloc[i].astype(str))
                    content += row + "\n"

                if len(self.data) > display_rows:
                    content += f"\n... и еще {len(self.data) - display_rows} строк ..."

                self.ui.content_text_edit.setText(content)

            else:
                # Для JSON или других форматов показываем первые строки DataFrame
                content = self.data.head(display_rows).to_string()
                if len(self.data) > display_rows:
                    content += f"\n... и еще {len(self.data) - display_rows} строк ..."

                self.ui.content_text_edit.setText(content)

        except Exception as e:
            self.ui.content_text_edit.setText(f"Ошибка при отображении данных: {str(e)}")

    def select_delimiter(self, delimiter):
        """Выбор разделителя для CSV файла"""
        if not self.filename.endswith('.csv'):
            QMessageBox.warning(self, "Предупреждение",
                                "Разделитель применяется только к CSV файлам!")
            return

        self.ui.delimiter_edit.setText(delimiter)

        # Сбрасываем выделение всех кнопок
        self.ui.comma_button.setChecked(delimiter == ",")
        self.ui.semicolon_button.setChecked(delimiter == ";")
        self.ui.tab_button.setChecked(delimiter == "\t")

    def select_delimiter_ui(self, delimiter):
        """Визуальный выбор разделителя"""
        self.ui.comma_button.setChecked(delimiter == ",")
        self.ui.semicolon_button.setChecked(delimiter == ";")
        self.ui.tab_button.setChecked(delimiter == "\t")

    def apply_delimiter(self):
        """Применение выбранного разделителя"""
        if not self.filename.endswith('.csv'):
            QMessageBox.warning(self, "Предупреждение",
                                "Разделитель применяется только к CSV файлам!")
            return

        try:
            delimiter = self.ui.delimiter_edit.text()
            if not delimiter:
                QMessageBox.warning(self, "Предупреждение", "Введите разделитель!")
                return

            # Перезагружаем данные с новым разделителем
            self.data = pd.read_csv(self.file_path, delimiter=delimiter, encoding='utf-8')
            self.current_delimiter = delimiter

            # Помечаем, что данные были изменены
            self.data_changed = True

            # Обновляем отображение
            self.display_file_info()
            self.display_file_content()

            # Обновляем кнопки
            self.select_delimiter_ui(delimiter)

            QMessageBox.information(self, "Успех", f"Разделитель '{delimiter}' успешно применен!")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось применить разделитель: {str(e)}")

    def go_back(self):
        """Переход на предыдущую страницу"""
        current_index = self.ui.stackedWidget.currentIndex()
        if current_index > 0:
            # Если мы на странице пропусков и открыта конкретная обработка
            if current_index == 2:  # Страница обработки пропусков
                if self.ui.pass_numeric_frame.isVisible() or self.ui.pass_categorical_frame.isVisible():
                    self.go_back_to_missing_selection()
                    return
            # Если мы на странице замены и открыта конкретная замена
            elif current_index == 3:  # Страница замены данных
                if self.ui.replace_numeric_frame.isVisible() or self.ui.replace_categorical_frame.isVisible():
                    self.go_back_to_replace_selection()
                    return

            self.ui.stackedWidget.setCurrentIndex(current_index - 1)

        # Обновляем состояние кнопок
        self.update_navigation_buttons()

    def go_next(self):
        """Переход на следующую страницу"""
        current_index = self.ui.stackedWidget.currentIndex()
        max_index = self.ui.stackedWidget.count() - 1

        if current_index < max_index:
            # Загружаем данные для следующей страницы
            self.load_page_data(current_index + 1)

            self.ui.stackedWidget.setCurrentIndex(current_index + 1)

        # Обновляем состояние кнопок
        self.update_navigation_buttons()

    def go_back_to_missing_selection(self):
        """Возврат к выбору типа обработки пропусков"""
        self.ui.pass_selection_frame.setVisible(True)
        self.ui.pass_numeric_frame.setVisible(False)
        self.ui.pass_categorical_frame.setVisible(False)

    def go_back_to_replace_selection(self):
        """Возврат к выбору типа замены данных"""
        self.ui.replace_selection_frame.setVisible(True)
        self.ui.replace_numeric_frame.setVisible(False)
        self.ui.replace_categorical_frame.setVisible(False)

    def load_page_data(self, page_index):
        """Загрузка данных для конкретной страницы"""
        if self.data is None:
            return

        if page_index == 1:  # Страница анализа столбцов
            self.load_analysis_page()
        elif page_index == 2:  # Страница обработки пропусков
            self.load_missing_page()
        elif page_index == 3:  # Страница замены данных
            self.load_replace_page()
        elif page_index == 4:  # Страница обработки дубликатов
            self.load_duplicates_page()
        elif page_index == 5:  # Страница удаления пропусков
            self.load_remove_missing_page()
        elif page_index == 6:  # Страница управления столбцами
            self.load_column_operations_page()

    def load_analysis_page(self):
        """Загрузка данных для страницы анализа столбцов"""
        try:
            # Обновляем информацию о наборе данных
            info_text = f"""
             Файл: {self.filename}
             Строк: {len(self.data):,}
             Столбцов: {len(self.data.columns)}
            """
            self.ui.dataset_info_label.setText(info_text)

            # Заполняем комбобокс с названиями столбцов
            self.ui.column_combo.clear()
            self.ui.column_combo.addItems(self.data.columns)

            # Подключаем обработчик изменения выбора столбца
            self.ui.column_combo.currentTextChanged.connect(self.on_column_selected)

            # Выбираем первый столбец по умолчанию
            if self.ui.column_combo.count() > 0:
                self.ui.column_combo.setCurrentIndex(0)
                self.on_column_selected(self.ui.column_combo.currentText())

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при загрузке страницы анализа: {str(e)}")

    def on_column_selected(self, column_name):
        """Обработчик выбора столбца"""
        try:
            if column_name not in self.data.columns:
                return

            column_data = self.data[column_name]

            # Обновляем статистику столбца
            dtype = str(column_data.dtype)
            unique_count = column_data.nunique()
            missing_count = column_data.isnull().sum()
            total_count = len(column_data)

            stats_text = f"""
            Тип данных: {dtype}
            Уникальных значений: {unique_count:,}
            Пустых значений: {missing_count:,} ({missing_count / total_count * 100:.1f}%)
            Всего значений: {total_count:,}
            """
            self.ui.column_stats_label.setText(stats_text)

            # Показываем уникальные значения
            unique_values = column_data.dropna().unique()
            display_values = unique_values[:50]  # Ограничиваем количество

            values_text = "\n".join(str(val) for val in display_values)
            self.ui.unique_values_text.setText(values_text)

            # Обновляем счетчик значений
            if len(unique_values) > 50:
                self.ui.values_count_label.setText(
                    f"Показано: 50 из {len(unique_values):,} значений")
            else:
                self.ui.values_count_label.setText(
                    f"Показано: {len(unique_values)} из {len(unique_values):,} значений")

            # Определяем рекомендуемый тип данных
            self.suggest_data_type(column_data, dtype)

        except Exception as e:
            print(f"Ошибка при обновлении информации о столбце: {str(e)}")

    def suggest_data_type(self, column_data, current_dtype):
        """Определение рекомендуемого типа данных"""
        try:
            # Сбрасываем выделение
            for i in range(self.ui.dtype_listwidget.count()):
                item = self.ui.dtype_listwidget.item(i)
                item.setSelected(False)

            # Определяем тип данных
            if pd.api.types.is_numeric_dtype(column_data):
                if pd.api.types.is_integer_dtype(column_data):
                    # Выбираем Integer
                    for i in range(self.ui.dtype_listwidget.count()):
                        item = self.ui.dtype_listwidget.item(i)
                        if "Integer" in item.text():
                            item.setSelected(True)
                            break
                else:
                    # Выбираем Float
                    for i in range(self.ui.dtype_listwidget.count()):
                        item = self.ui.dtype_listwidget.item(i)
                        if "Float" in item.text():
                            item.setSelected(True)
                            break
            else:
                # Выбираем Object
                for i in range(self.ui.dtype_listwidget.count()):
                    item = self.ui.dtype_listwidget.item(i)
                    if "Object" in item.text():
                        item.setSelected(True)
                        break

        except Exception as e:
            print(f"Ошибка при определении типа данных: {str(e)}")

    def auto_detect_data_type(self):
        """Автоматическое определение типа данных"""
        try:
            column_name = self.ui.column_combo.currentText()
            if not column_name:
                return

            column_data = self.data[column_name]
            self.suggest_data_type(column_data, str(column_data.dtype))

            QMessageBox.information(self, "Автоопределение",
                                    "Тип данных автоматически определен!")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при автоопределении: {str(e)}")

    def apply_data_type(self):
        """Применение выбранного типа данных"""
        try:
            column_name = self.ui.column_combo.currentText()
            if not column_name:
                QMessageBox.warning(self, "Предупреждение", "Выберите столбец!")
                return

            # Получаем выбранный тип
            selected_items = self.ui.dtype_listwidget.selectedItems()
            if not selected_items:
                QMessageBox.warning(self, "Предупреждение", "Выберите тип данных!")
                return

            selected_type = selected_items[0].text()

            # Применяем тип данных
            if "Integer" in selected_type:
                self.data[column_name] = pd.to_numeric(self.data[column_name],
                                                       errors='coerce').astype('Int64')
                new_type = "integer"
            elif "Float" in selected_type:
                self.data[column_name] = pd.to_numeric(self.data[column_name],
                                                       errors='coerce').astype('float64')
                new_type = "float"
            elif "Object" in selected_type:
                self.data[column_name] = self.data[column_name].astype('object')
                new_type = "object"
            else:
                QMessageBox.warning(self, "Предупреждение", "Неизвестный тип данных!")
                return

            # Помечаем, что данные были изменены
            self.data_changed = True

            # Обновляем информацию о столбце
            self.on_column_selected(column_name)

            QMessageBox.information(self, "Успех",
                                    f"Тип данных столбца '{column_name}' изменен на {new_type}!")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при изменении типа данных: {str(e)}")

    def load_missing_page(self):
        """Загрузка данных для страницы обработки пропусков"""
        try:
            # Заполняем комбобокс с названиями столбцов
            self.ui.pass_column_combo.clear()
            self.ui.pass_column_combo.addItems(self.data.columns)

            # Выбираем первый столбец по умолчанию
            if self.ui.pass_column_combo.count() > 0:
                self.ui.pass_column_combo.setCurrentIndex(0)
                self.update_missing_stats(self.ui.pass_column_combo.currentText())

            # Сбрасываем видимость фреймов
            self.reset_ui_state()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка",
                                 f"Ошибка при загрузке страницы пропусков: {str(e)}")

    def update_missing_stats(self, column_name):
        """Обновление статистики пропусков"""
        try:
            if column_name not in self.data.columns:
                return

            column_data = self.data[column_name]
            missing_count = column_data.isnull().sum()
            total_count = len(column_data)
            percentage = (missing_count / total_count * 100) if total_count > 0 else 0

            # Определяем тип столбца
            dtype = str(column_data.dtype)
            if pd.api.types.is_numeric_dtype(column_data):
                column_type = "Количественный"
                self.ui.pass_numeric_button.setEnabled(missing_count > 0)
                self.ui.pass_categorical_button.setEnabled(False)
            else:
                column_type = "Категориальный"
                self.ui.pass_numeric_button.setEnabled(False)
                self.ui.pass_categorical_button.setEnabled(missing_count > 0)

            # Обновляем информацию
            self.ui.pass_column_type_label.setText(f"Тип: {column_type}")
            self.ui.pass_stats_label.setText(
                f"Пропусков: {missing_count:,}/{total_count:,} ({percentage:.1f}%)")

        except Exception as e:
            print(f"Ошибка при обновлении статистики пропусков: {str(e)}")

    def open_numeric_missing(self):
        """Открытие секции обработки числовых пропусков"""
        column_name = self.ui.pass_column_combo.currentText()
        if column_name and self.ui.pass_numeric_button.isEnabled():
            self.ui.pass_selection_frame.setVisible(False)
            self.ui.pass_numeric_frame.setVisible(True)
            self.load_numeric_missing_data(column_name)

    def open_categorical_missing(self):
        """Открытие секции обработки категориальных пропусков"""
        column_name = self.ui.pass_column_combo.currentText()
        if column_name and self.ui.pass_categorical_button.isEnabled():
            self.ui.pass_selection_frame.setVisible(False)
            self.ui.pass_categorical_frame.setVisible(True)
            self.load_categorical_missing_data(column_name)

    def load_numeric_missing_data(self, column_name):
        """Загрузка данных для обработки числовых пропусков"""
        try:
            if column_name not in self.data.columns:
                return

            column_data = self.data[column_name]
            missing_count = column_data.isnull().sum()
            total_count = len(column_data)
            percentage = (missing_count / total_count * 100) if total_count > 0 else 0

            # Обновляем информацию
            info_text = f"""
            Столбец: {column_name}
            Тип: {str(column_data.dtype)}
            Пропусков: {missing_count:,} ({percentage:.1f}%)
            Всего значений: {total_count:,}
            """
            self.ui.pass_numeric_info.setText(info_text)

            # Заполняем таблицу пропущенных значений
            self.load_missing_values_table(column_name)

            # Показываем статистику
            if pd.api.types.is_numeric_dtype(column_data):
                stats_text = f"""
                Мин: {column_data.min():.2f}
                Макс: {column_data.max():.2f}
                Среднее: {column_data.mean():.2f}
                Медиана: {column_data.median():.2f}
                Стандартное отклонение: {column_data.std():.2f}
                """
                self.ui.pass_numeric_stats_text.setText(stats_text)

            # Заполняем методы обработки
            self.ui.pass_numeric_methods_list.clear()
            methods = [
                "Среднее значение (mean)",
                "Медиана (median)",
                "Мода (mode)",
                "Константа (0)",
                "Константа (1)",
                "Предыдущее значение (forward fill)",
                "Следующее значение (backward fill)",
                "Линейная интерполяция",
                "Свое значение"
            ]
            self.ui.pass_numeric_methods_list.addItems(methods)
            self.ui.pass_numeric_methods_list.setCurrentRow(0)

            # Подключаем выбор метода
            self.ui.pass_numeric_methods_list.currentRowChanged.connect(
                self.on_numeric_method_selected)

        except Exception as e:
            print(f"Ошибка при загрузке данных числовых пропусков: {str(e)}")

    def load_missing_values_table(self, column_name):
        """Загрузка таблицы с пропущенными значениями"""
        try:
            # Находим строки с пропусками
            missing_indices = self.data[self.data[column_name].isnull()].index.tolist()

            # Показываем первые 20 строк с пропусками
            display_indices = missing_indices[:20]
            display_data = self.data.loc[display_indices].copy()

            # Устанавливаем таблицу
            self.ui.pass_numeric_table.setRowCount(len(display_data))
            self.ui.pass_numeric_table.setColumnCount(len(self.data.columns))
            self.ui.pass_numeric_table.setHorizontalHeaderLabels(self.data.columns.tolist())

            # Заполняем таблицу
            for i, (idx, row) in enumerate(display_data.iterrows()):
                for j, col in enumerate(self.data.columns):
                    value = row[col]
                    if pd.isna(value):
                        item_text = "NaN"
                        item = QTableWidgetItem(item_text)
                        self.ui.pass_numeric_table.setItem(i, j, item)
                        item.setBackground(Qt.GlobalColor.yellow)
                    else:
                        item_text = str(value)
                        item = QTableWidgetItem(item_text)
                        self.ui.pass_numeric_table.setItem(i, j, item)
                        item.setBackground(Qt.GlobalColor.white)

            # Настраиваем заголовки
            header = self.ui.pass_numeric_table.horizontalHeader()
            header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        except Exception as e:
            print(f"Ошибка при загрузке таблицы пропусков: {str(e)}")

    def on_numeric_method_selected(self, row):
        """Обработчик выбора метода обработки числовых пропусков"""
        if row == 8:  # "Свое значение"
            self.ui.pass_custom_numeric_edit.setEnabled(True)
            self.ui.pass_custom_numeric_edit.setFocus()
        else:
            self.ui.pass_custom_numeric_edit.setEnabled(False)
            self.ui.pass_custom_numeric_edit.clear()

    def apply_numeric_missing(self):
        """Применение обработки числовых пропусков"""
        try:
            column_name = self.ui.pass_column_combo.currentText()
            if not column_name:
                QMessageBox.warning(self, "Предупреждение", "Выберите столбец!")
                return

            column_data = self.data[column_name]
            missing_before = column_data.isnull().sum()

            if missing_before == 0:
                QMessageBox.information(self, "Информация", "В выбранном столбце нет пропусков!")
                return

            # Получаем выбранный метод
            selected_row = self.ui.pass_numeric_methods_list.currentRow()

            # Применяем метод
            if selected_row == 0:  # Среднее
                fill_value = column_data.mean()
                self.data[column_name] = column_data.fillna(fill_value)
                method_name = "среднее значение"

            elif selected_row == 1:  # Медиана
                fill_value = column_data.median()
                self.data[column_name] = column_data.fillna(fill_value)
                method_name = "медиана"

            elif selected_row == 2:  # Мода
                fill_value = column_data.mode()[0] if not column_data.mode().empty else 0
                self.data[column_name] = column_data.fillna(fill_value)
                method_name = "мода"

            elif selected_row == 3:  # Константа 0
                self.data[column_name] = column_data.fillna(0)
                method_name = "константа 0"

            elif selected_row == 4:  # Константа 1
                self.data[column_name] = column_data.fillna(1)
                method_name = "константа 1"

            elif selected_row == 5:  # Forward fill
                self.data[column_name] = column_data.fillna(method='ffill')
                method_name = "предыдущее значение"

            elif selected_row == 6:  # Backward fill
                self.data[column_name] = column_data.fillna(method='bfill')
                method_name = "следующее значение"

            elif selected_row == 7:  # Интерполяция
                self.data[column_name] = column_data.interpolate()
                method_name = "линейная интерполяция"

            elif selected_row == 8:  # Свое значение
                custom_value = self.ui.pass_custom_numeric_edit.text()
                if not custom_value:
                    QMessageBox.warning(self, "Предупреждение", "Введите значение!")
                    return
                try:
                    fill_value = float(custom_value)
                    self.data[column_name] = column_data.fillna(fill_value)
                    method_name = f"значение {fill_value}"
                except ValueError:
                    QMessageBox.warning(self, "Ошибка", "Введите числовое значение!")
                    return
            else:
                QMessageBox.warning(self, "Предупреждение", "Выберите метод обработки!")
                return

            missing_after = self.data[column_name].isnull().sum()

            # Помечаем, что данные были изменены
            self.data_changed = True

            QMessageBox.information(self, "Успех",
                                    f"Обработано пропусков: {missing_before - missing_after}\n"
                                    f"Метод: {method_name}\n"
                                    f"Осталось пропусков: {missing_after}")

            # Возвращаемся к выбору
            self.go_back_to_missing_selection()
            self.update_missing_stats(column_name)

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при обработке пропусков: {str(e)}")

    def load_categorical_missing_data(self, column_name):
        """Загрузка данных для обработки категориальных пропусков"""
        try:
            if column_name not in self.data.columns:
                return

            column_data = self.data[column_name]
            missing_count = column_data.isnull().sum()
            total_count = len(column_data)
            percentage = (missing_count / total_count * 100) if total_count > 0 else 0

            # Обновляем информацию
            info_text = f"""
            Столбец: {column_name}
            Тип: {str(column_data.dtype)}
            Пропусков: {missing_count:,} ({percentage:.1f}%)
            Всего значений: {total_count:,}
            """
            self.ui.pass_categorical_info.setText(info_text)

            # Показываем уникальные значения
            unique_values = column_data.dropna().unique()
            values_text = "\n".join(str(val) for val in unique_values[:50])
            self.ui.pass_categories_text.setText(values_text)

            # Показываем статистику
            if not column_data.empty:
                mode_value = column_data.mode()
                mode_text = mode_value[0] if not mode_value.empty else "нет"
                mode_count = (column_data == mode_text).sum() if not mode_value.empty else 0

                stats_text = f"""
                Всего значений: {total_count:,}
                Уникальных: {len(unique_values):,}
                Самое частое: {mode_text}
                Встречается: {mode_count:,} раз ({mode_count / total_count * 100:.1f}%)
                """
                self.ui.pass_categorical_stats.setText(stats_text)

            # Создаем и отображаем круговую диаграмму
            self.create_pie_chart(column_name)

        except Exception as e:
            print(f"Ошибка при загрузке данных категориальных пропусков: {str(e)}")

    def create_pie_chart(self, column_name):
        """Создание круговой диаграммы для категориальных данных"""
        try:
            if column_name not in self.data.columns:
                return

            column_data = self.data[column_name]

            # Подсчет значений
            value_counts = column_data.value_counts()

            # Берем топ-10 значений для отображения
            if len(value_counts) > 10:
                top_values = value_counts.head(10)
                other_count = value_counts.iloc[10:].sum()
                if other_count > 0:
                    top_values['Другие'] = other_count
            else:
                top_values = value_counts

            # Создаем временный файл для диаграммы
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            self.temp_files.append(temp_file.name)

            # Создаем диаграмму
            plt.figure(figsize=(8, 6))
            plt.pie(top_values.values, labels=top_values.index, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title(f'Распределение значений: {column_name}')

            # Сохраняем диаграмму во временный файл
            plt.tight_layout()
            plt.savefig(temp_file.name, dpi=100)
            plt.close()

            # Отображаем диаграмму в QLabel
            pixmap = QPixmap(temp_file.name)
            self.ui.pass_chart_placeholder.setPixmap(pixmap.scaled(
                self.ui.pass_chart_placeholder.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))

        except Exception as e:
            print(f"Ошибка при создании круговой диаграммы: {str(e)}")
            self.ui.pass_chart_placeholder.setText("Не удалось создать диаграмму")

    def apply_categorical_missing(self):
        """Применение обработки категориальных пропусков"""
        try:
            column_name = self.ui.pass_column_combo.currentText()
            if not column_name:
                QMessageBox.warning(self, "Предупреждение", "Выберите столбец!")
                return

            column_data = self.data[column_name]
            missing_before = column_data.isnull().sum()

            if missing_before == 0:
                QMessageBox.information(self, "Информация", "В выбранном столбце нет пропусков!")
                return

            # Получаем значение для заполнения
            fill_value = self.ui.pass_custom_category_edit.text()
            if not fill_value:
                QMessageBox.warning(self, "Предупреждение",
                                    "Введите значение для заполнения пропусков!")
                return

            # Заполняем пропуски
            self.data[column_name] = column_data.fillna(fill_value)
            missing_after = self.data[column_name].isnull().sum()

            # Помечаем, что данные были изменены
            self.data_changed = True

            QMessageBox.information(self, "Успех",
                                    f"Обработано пропусков: {missing_before - missing_after}\n"
                                    f"Заполнено значением: '{fill_value}'\n"
                                    f"Осталось пропусков: {missing_after}")

            # Возвращаемся к выбору
            self.go_back_to_missing_selection()
            self.update_missing_stats(column_name)

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при обработке пропусков: {str(e)}")

    def load_replace_page(self):
        """Загрузка данных для страницы замены данных"""
        try:
            # Заполняем список столбцов
            self.ui.replace_column_listwidget.clear()
            self.ui.replace_column_listwidget.addItems(self.data.columns)

            # Сбрасываем видимость фреймов
            self.reset_ui_state()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка",
                                 f"Ошибка при загрузке страницы замены: {str(e)}")

    def on_replace_column_selected(self):
        """Обработчик выбора столбца для замены"""
        selected_items = self.ui.replace_column_listwidget.selectedItems()
        if not selected_items:
            self.ui.replace_selected_column_name.setText("Не выбран")
            self.ui.replace_selected_column_type.setText("...")
            self.ui.replace_numeric_button.setEnabled(False)
            self.ui.replace_categorical_button.setEnabled(False)
            return

        column_name = selected_items[0].text()
        if column_name not in self.data.columns:
            return

        column_data = self.data[column_name]
        dtype = str(column_data.dtype)

        self.ui.replace_selected_column_name.setText(column_name)
        self.ui.replace_selected_column_type.setText(dtype)

        # Активируем соответствующую кнопку в зависимости от типа данных
        if pd.api.types.is_numeric_dtype(column_data):
            self.ui.replace_numeric_button.setEnabled(True)
            self.ui.replace_categorical_button.setEnabled(False)
        else:
            self.ui.replace_numeric_button.setEnabled(False)
            self.ui.replace_categorical_button.setEnabled(True)

    def open_numeric_replace(self):
        """Открытие секции замены числовых данных"""
        selected_items = self.ui.replace_column_listwidget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Предупреждение", "Выберите столбец!")
            return

        column_name = selected_items[0].text()
        if column_name:
            self.ui.replace_selection_frame.setVisible(False)
            self.ui.replace_numeric_frame.setVisible(True)
            self.load_numeric_replace_data(column_name)

    def open_categorical_replace(self):
        """Открытие секции замены категориальных данных"""
        selected_items = self.ui.replace_column_listwidget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Предупреждение", "Выберите столбец!")
            return

        column_name = selected_items[0].text()
        if column_name:
            self.ui.replace_selection_frame.setVisible(False)
            self.ui.replace_categorical_frame.setVisible(True)
            self.load_categorical_replace_data(column_name)

    def load_numeric_replace_data(self, column_name):
        """Загрузка данных для замены числовых значений"""
        try:
            if column_name not in self.data.columns:
                return

            column_data = self.data[column_name]

            # Обновляем информацию о столбце
            self.ui.replace_numeric_current_name.setText(column_name)

            # Показываем статистику
            if pd.api.types.is_numeric_dtype(column_data):
                stats_text = f"""
                Мин: {column_data.min():.2f}
                Макс: {column_data.max():.2f}
                Среднее: {column_data.mean():.2f}
                Медиана: {column_data.median():.2f}
                Стандартное отклонение: {column_data.std():.2f}
                """
                self.ui.replace_numeric_stats_text.setText(stats_text)

            # Создаем и отображаем boxplot
            self.create_boxplot(column_name)

            # Устанавливаем начальные значения
            self.ui.replace_numeric_radio_greater.setChecked(True)
            self.ui.replace_numeric_radio_multiply.setChecked(True)
            self.ui.replace_numeric_multiply_edit.setText("1.0")

            # Обновляем превью
            self.update_replace_numeric_preview()

        except Exception as e:
            print(f"Ошибка при загрузке данных числовой замены: {str(e)}")

    def create_boxplot(self, column_name):
        """Создание boxplot для числовых данных"""
        try:
            if column_name not in self.data.columns:
                return

            column_data = self.data[column_name].dropna()

            if len(column_data) == 0:
                self.ui.replace_numeric_boxplot_placeholder.setText(
                    "Нет данных для построения графика")
                return

            # Создаем временный файл для boxplot
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            self.temp_files.append(temp_file.name)

            # Создаем boxplot
            plt.figure(figsize=(8, 6))
            plt.boxplot(column_data, vert=True, patch_artist=True,
                        boxprops=dict(facecolor='lightblue'),
                        medianprops=dict(color='red', linewidth=2))

            # Добавляем точки данных
            y = column_data
            x = np.random.normal(1, 0.04, size=len(y))
            plt.plot(x, y, 'r.', alpha=0.4)

            plt.title(f'Box Plot: {column_name}')
            plt.ylabel('Значения')
            plt.grid(True, alpha=0.3)

            # Добавляем статистику
            stats_text = f"""Статистика:
            Мин: {column_data.min():.2f}
            Q1: {column_data.quantile(0.25):.2f}
            Медиана: {column_data.median():.2f}
            Q3: {column_data.quantile(0.75):.2f}
            Макс: {column_data.max():.2f}"""

            plt.figtext(0.02, 0.02, stats_text, fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

            # Сохраняем график во временный файл
            plt.tight_layout()
            plt.savefig(temp_file.name, dpi=100)
            plt.close()

            # Отображаем график в QLabel
            pixmap = QPixmap(temp_file.name)
            self.ui.replace_numeric_boxplot_placeholder.setPixmap(pixmap.scaled(
                self.ui.replace_numeric_boxplot_placeholder.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))

        except Exception as e:
            print(f"Ошибка при создании boxplot: {str(e)}")
            self.ui.replace_numeric_boxplot_placeholder.setText("Не удалось создать график")

    def update_replace_numeric_preview(self):
        """Обновление предпросмотра для числовой замены"""
        try:
            column_name = self.ui.replace_numeric_current_name.text()
            if not column_name or column_name == "...":
                return

            column_data = self.data[column_name]

            # Получаем настройки
            threshold_text = self.ui.replace_numeric_threshold_edit.text()
            if not threshold_text:
                self.ui.replace_numeric_preview_info_label.setText("Введите пороговое значение")
                self.clear_preview_table(self.ui.replace_numeric_preview_table)
                return

            try:
                threshold = float(threshold_text)
            except ValueError:
                self.ui.replace_numeric_preview_info_label.setText(
                    "Некорректное пороговое значение")
                self.clear_preview_table(self.ui.replace_numeric_preview_table)
                return

            # Определяем условие
            if self.ui.replace_numeric_radio_greater.isChecked():
                condition = column_data > threshold
                condition_text = f"> {threshold}"
            elif self.ui.replace_numeric_radio_less.isChecked():
                condition = column_data < threshold
                condition_text = f"< {threshold}"
            elif self.ui.replace_numeric_radio_greater_equal.isChecked():
                condition = column_data >= threshold
                condition_text = f"≥ {threshold}"
            elif self.ui.replace_numeric_radio_less_equal.isChecked():
                condition = column_data <= threshold
                condition_text = f"≤ {threshold}"
            else:
                return

            # Находим строки, соответствующие условию
            matching_indices = column_data[condition].index.tolist()
            matching_count = len(matching_indices)

            # Показываем первые 20 строк
            display_indices = matching_indices[:20]
            display_data = self.data.loc[display_indices].copy()

            # Заполняем таблицу превью
            self.ui.replace_numeric_preview_table.setRowCount(len(display_data))
            self.ui.replace_numeric_preview_table.setColumnCount(min(5, len(self.data.columns)))

            # Выбираем до 5 столбцов для отображения
            cols_to_show = [column_name] + [col for col in self.data.columns if col != column_name][
                                           :4]
            self.ui.replace_numeric_preview_table.setHorizontalHeaderLabels(cols_to_show)

            for i, (idx, row) in enumerate(display_data.iterrows()):
                for j, col in enumerate(cols_to_show):
                    value = row[col]
                    if pd.isna(value):
                        item_text = "NaN"
                        item = QTableWidgetItem(item_text)
                        item.setBackground(Qt.GlobalColor.yellow)
                    else:
                        item_text = str(value)
                        item = QTableWidgetItem(item_text)
                        if col == column_name:
                            item.setBackground(Qt.GlobalColor.lightGray)

                    self.ui.replace_numeric_preview_table.setItem(i, j, item)

            # Настраиваем заголовки
            header = self.ui.replace_numeric_preview_table.horizontalHeader()
            header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

            # Обновляем информацию
            self.ui.replace_numeric_preview_info_label.setText(
                f"Найдено строк: {matching_count} (условие: {condition_text})"
            )

        except Exception as e:
            print(f"Ошибка при обновлении превью: {str(e)}")

    def clear_preview_table(self, table_widget):
        """Очистка таблицы предпросмотра"""
        table_widget.setRowCount(0)
        table_widget.setColumnCount(0)

    def apply_numeric_replace(self):
        """Применение замены числовых данных"""
        try:
            column_name = self.ui.replace_numeric_current_name.text()
            if not column_name or column_name == "...":
                QMessageBox.warning(self, "Предупреждение", "Не выбран столбец!")
                return

            column_data = self.data[column_name]

            # Получаем настройки
            threshold_text = self.ui.replace_numeric_threshold_edit.text()
            if not threshold_text:
                QMessageBox.warning(self, "Предупреждение", "Введите пороговое значение!")
                return

            try:
                threshold = float(threshold_text)
            except ValueError:
                QMessageBox.warning(self, "Предупреждение", "Введите корректное число для порога!")
                return

            # Определяем условие
            if self.ui.replace_numeric_radio_greater.isChecked():
                condition = column_data > threshold
                condition_text = f"> {threshold}"
            elif self.ui.replace_numeric_radio_less.isChecked():
                condition = column_data < threshold
                condition_text = f"< {threshold}"
            elif self.ui.replace_numeric_radio_greater_equal.isChecked():
                condition = column_data >= threshold
                condition_text = f"≥ {threshold}"
            elif self.ui.replace_numeric_radio_less_equal.isChecked():
                condition = column_data <= threshold
                condition_text = f"≤ {threshold}"
            else:
                QMessageBox.warning(self, "Предупреждение", "Выберите условие замены!")
                return

            # Определяем операцию
            if self.ui.replace_numeric_radio_multiply.isChecked():
                multiply_text = self.ui.replace_numeric_multiply_edit.text()
                if not multiply_text:
                    QMessageBox.warning(self, "Предупреждение", "Введите множитель!")
                    return

                try:
                    multiplier = float(multiply_text)
                except ValueError:
                    QMessageBox.warning(self, "Предупреждение",
                                        "Введите корректное число для множителя!")
                    return

                # Применяем умножение
                rows_affected = condition.sum()
                self.data.loc[condition, column_name] = column_data[condition] * multiplier
                operation_text = f"умножено на {multiplier}"

            elif self.ui.replace_numeric_radio_delete.isChecked():
                # Удаляем строки
                rows_affected = condition.sum()
                self.data = self.data[~condition].reset_index(drop=True)
                operation_text = "удалены строки"

            else:
                QMessageBox.warning(self, "Предупреждение", "Выберите операцию замены!")
                return

            # Помечаем, что данные были изменены
            self.data_changed = True

            QMessageBox.information(self, "Успех",
                                    f"Замена выполнена успешно!\n\n"
                                    f"Столбец: {column_name}\n"
                                    f"Условие: {condition_text}\n"
                                    f"Операция: {operation_text}\n"
                                    f"Затронуто строк: {rows_affected}")

            # Обновляем данные
            self.load_numeric_replace_data(column_name)

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при замене данных: {str(e)}")

    def load_categorical_replace_data(self, column_name):
        """Загрузка данных для замены категориальных значений"""
        try:
            if column_name not in self.data.columns:
                return

            column_data = self.data[column_name]

            # Обновляем информацию о столбце
            self.ui.replace_categorical_current_name.setText(column_name)

            # Заполняем список уникальных значений
            self.ui.replace_categorical_values_list.clear()
            unique_values = column_data.dropna().unique()
            for value in unique_values[:100]:  # Ограничиваем количество
                self.ui.replace_categorical_values_list.addItem(str(value))

            self.ui.replace_categorical_values_info_label.setText(
                f"Уникальных значений: {len(unique_values)}")

            # Показываем статистику
            if not column_data.empty:
                mode_value = column_data.mode()
                mode_text = mode_value[0] if not mode_value.empty else "нет"
                mode_count = (column_data == mode_text).sum() if not mode_value.empty else 0
                total_count = len(column_data)

                stats_text = f"""
                Всего значений: {total_count:,}
                Уникальных: {len(unique_values):,}
                Самое частое: {mode_text}
                Встречается: {mode_count:,} раз ({mode_count / total_count * 100:.1f}%)
                """
                self.ui.replace_categorical_stats_text.setText(stats_text)

            # Очищаем поля
            self.ui.replace_categorical_search_edit.clear()
            self.ui.replace_categorical_replace_with_edit.clear()
            self.clear_preview_table(self.ui.replace_categorical_preview_table)

        except Exception as e:
            print(f"Ошибка при загрузке данных категориальной замены: {str(e)}")

    def filter_categorical_values(self):
        """Фильтрация значений в списке категориальных данных"""
        search_text = self.ui.replace_categorical_search_edit.text().lower()

        # Сохраняем выбранный элемент
        selected_items = self.ui.replace_categorical_values_list.selectedItems()
        selected_value = selected_items[0].text() if selected_items else None

        # Показываем все элементы если поиск пустой
        for i in range(self.ui.replace_categorical_values_list.count()):
            item = self.ui.replace_categorical_values_list.item(i)
            if not search_text or search_text in item.text().lower():
                item.setHidden(False)
            else:
                item.setHidden(True)

        # Восстанавливаем выбор если элемент видим
        if selected_value:
            items = self.ui.replace_categorical_values_list.findItems(selected_value,
                                                                      Qt.MatchFlag.MatchExactly)
            if items and not items[0].isHidden():
                items[0].setSelected(True)

    def update_categorical_preview(self):
        """Обновление предпросмотра для категориальной замены"""
        try:
            selected_items = self.ui.replace_categorical_values_list.selectedItems()
            if not selected_items:
                self.clear_preview_table(self.ui.replace_categorical_preview_table)
                self.ui.replace_categorical_preview_info_label.setText(
                    "Выберите значение для замены")
                return

            old_value = selected_items[0].text()
            new_value = self.ui.replace_categorical_replace_with_edit.text()

            if not new_value:
                self.ui.replace_categorical_preview_info_label.setText("Введите новое значение")
                self.clear_preview_table(self.ui.replace_categorical_preview_table)
                return

            column_name = self.ui.replace_categorical_current_name.text()
            column_data = self.data[column_name]

            # Находим строки с выбранным значением
            matching_indices = column_data[column_data == old_value].index.tolist()
            matching_count = len(matching_indices)

            # Показываем первые 20 строк
            display_indices = matching_indices[:20]
            display_data = self.data.loc[display_indices].copy()

            # Заполняем таблицу превью
            self.ui.replace_categorical_preview_table.setRowCount(len(display_data))
            self.ui.replace_categorical_preview_table.setColumnCount(min(5, len(self.data.columns)))

            # Выбираем до 5 столбцов для отображения
            cols_to_show = [column_name] + [col for col in self.data.columns if col != column_name][
                                           :4]
            self.ui.replace_categorical_preview_table.setHorizontalHeaderLabels(cols_to_show)

            for i, (idx, row) in enumerate(display_data.iterrows()):
                for j, col in enumerate(cols_to_show):
                    value = row[col]
                    if pd.isna(value):
                        item_text = "NaN"
                        item = QTableWidgetItem(item_text)
                        item.setBackground(Qt.GlobalColor.yellow)
                    else:
                        item_text = str(value)
                        item = QTableWidgetItem(item_text)
                        if col == column_name:
                            item.setBackground(Qt.GlobalColor.lightGray)
                            if value == old_value:
                                item.setForeground(Qt.GlobalColor.red)

                    self.ui.replace_categorical_preview_table.setItem(i, j, item)

            # Настраиваем заголовки
            header = self.ui.replace_categorical_preview_table.horizontalHeader()
            header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

            # Обновляем информацию
            self.ui.replace_categorical_preview_info_label.setText(
                f"Найдено строк: {matching_count} | Замена: '{old_value}' → '{new_value}'"
            )

        except Exception as e:
            print(f"Ошибка при обновлении превью: {str(e)}")

    def apply_categorical_replace(self):
        """Применение замены категориальных данных"""
        try:
            selected_items = self.ui.replace_categorical_values_list.selectedItems()
            if not selected_items:
                QMessageBox.warning(self, "Предупреждение", "Выберите значение для замены!")
                return

            old_value = selected_items[0].text()
            new_value = self.ui.replace_categorical_replace_with_edit.text()

            if not new_value:
                QMessageBox.warning(self, "Предупреждение", "Введите новое значение!")
                return

            column_name = self.ui.replace_categorical_current_name.text()
            column_data = self.data[column_name]

            # Подсчитываем сколько строк будет заменено
            rows_affected = (column_data == old_value).sum()

            if rows_affected == 0:
                QMessageBox.information(self, "Информация",
                                        f"Значение '{old_value}' не найдено в столбце '{column_name}'")
                return

            # Подтверждение
            reply = QMessageBox.question(
                self, "Подтверждение замены",
                f"Заменить '{old_value}' на '{new_value}' в столбце '{column_name}'?\n\n"
                f"Будет заменено {rows_affected} строк.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.No:
                return

            # Выполняем замену
            self.data[column_name] = column_data.replace(old_value, new_value)

            # Помечаем, что данные были изменены
            self.data_changed = True

            QMessageBox.information(self, "Успех",
                                    f"Замена выполнена успешно!\n\n"
                                    f"Столбец: {column_name}\n"
                                    f"Замена: '{old_value}' → '{new_value}'\n"
                                    f"Затронуто строк: {rows_affected}")

            # Обновляем данные
            self.load_categorical_replace_data(column_name)

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при замене данных: {str(e)}")

    def load_duplicates_page(self):
        """Загрузка данных для страницы обработки дубликатов"""
        try:
            # Находим дубликаты
            duplicates = self.data[self.data.duplicated(keep=False)]

            if len(duplicates) == 0:
                self.ui.stats_label.setText("Дубликатов не найдено")
                self.ui.duplicates_table.setRowCount(0)
                self.ui.duplicates_table.setColumnCount(0)
                self.ui.table_info_label.setText("Дубликатов не найдено")
                self.ui.remove_duplicates_btn.setEnabled(False)
                self.ui.keep_duplicates_btn.setEnabled(False)
                return

            # Сортируем по всем столбцам для группировки дубликатов
            duplicates_sorted = duplicates.sort_values(by=list(duplicates.columns))

            # Отображаем статистику
            total_duplicates = len(duplicates)
            unique_duplicate_groups = len(duplicates_sorted.drop_duplicates())

            self.ui.stats_label.setText(
                f"Найдено {total_duplicates} дубликатов в {unique_duplicate_groups} группах"
            )

            # Заполняем таблицу
            self.ui.duplicates_table.setRowCount(len(duplicates_sorted))
            self.ui.duplicates_table.setColumnCount(len(self.data.columns))
            self.ui.duplicates_table.setHorizontalHeaderLabels(self.data.columns.tolist())

            for i, (idx, row) in enumerate(duplicates_sorted.iterrows()):
                for j, col in enumerate(self.data.columns):
                    value = row[col]
                    if pd.isna(value):
                        item_text = "NaN"
                        item = QTableWidgetItem(item_text)
                        item.setBackground(Qt.GlobalColor.yellow)
                    else:
                        item_text = str(value)
                        item = QTableWidgetItem(item_text)
                        # Подсвечиваем дубликаты
                        item.setBackground(Qt.GlobalColor.lightGray)

                    self.ui.duplicates_table.setItem(i, j, item)

            # Настраиваем заголовки
            header = self.ui.duplicates_table.horizontalHeader()
            header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

            # Обновляем информацию
            self.ui.table_info_label.setText(f"Показано {len(duplicates_sorted)} строк")

            # Активируем кнопки
            self.ui.remove_duplicates_btn.setEnabled(True)
            self.ui.keep_duplicates_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при загрузке дубликатов: {str(e)}")

    def remove_duplicates(self):
        """Удаление дубликатов"""
        try:
            # Подсчитываем количество дубликатов
            duplicates_count = self.data.duplicated().sum()

            if duplicates_count == 0:
                QMessageBox.information(self, "Информация", "Дубликатов не найдено")
                return

            # Подтверждение
            reply = QMessageBox.question(
                self, "Подтверждение удаления",
                f"Удалить {duplicates_count} дубликатов?\n\n"
                f"После удаления останется {len(self.data) - duplicates_count} уникальных строк.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.No:
                return

            # Сохраняем старый размер
            old_size = len(self.data)

            # Удаляем дубликаты, оставляя первую встреченную строку
            self.data = self.data.drop_duplicates().reset_index(drop=True)

            # Помечаем, что данные были изменены
            self.data_changed = True

            # Подсчитываем новые дубликаты
            new_duplicates = self.data.duplicated().sum()

            QMessageBox.information(self, "Успех",
                                    f"Дубликаты успешно удалены!\n\n"
                                    f"Было: {old_size} строк\n"
                                    f"Стало: {len(self.data)} строк\n"
                                    f"Удалено: {duplicates_count} дубликатов\n"
                                    f"Осталось дубликатов: {new_duplicates}")

            # Обновляем таблицу
            self.load_duplicates_page()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при удалении дубликатов: {str(e)}")

    def keep_duplicates(self):
        """Оставить только дубликаты"""
        try:
            # Находим дубликаты
            duplicates_mask = self.data.duplicated(keep=False)
            duplicates_only = self.data[duplicates_mask]

            if len(duplicates_only) == 0:
                QMessageBox.information(self, "Информация", "Дубликатов не найдено")
                return

            # Подтверждение
            reply = QMessageBox.question(
                self, "Подтверждение",
                f"Оставить только {len(duplicates_only)} дубликатов?\n\n"
                f"Будет удалено {len(self.data) - len(duplicates_only)} уникальных строк.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.No:
                return

            # Сохраняем старый размер
            old_size = len(self.data)

            # Оставляем только дубликаты
            self.data = duplicates_only.reset_index(drop=True)

            # Помечаем, что данные были изменены
            self.data_changed = True

            QMessageBox.information(self, "Успех",
                                    f"Оставлены только дубликаты!\n\n"
                                    f"Было: {old_size} строк\n"
                                    f"Стало: {len(self.data)} строк\n"
                                    f"Удалено: {old_size - len(self.data)} уникальных строк")

            # Обновляем таблицу
            self.load_duplicates_page()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при обработке дубликатов: {str(e)}")

    def load_remove_missing_page(self):
        """Загрузка данных для страницы удаления пропусков"""
        try:
            # Обновляем информацию о пропусках
            self.update_missing_info()

            # Заполняем комбобокс столбцами
            self.ui.column_combo_missing.clear()
            self.ui.column_combo_missing.addItems(self.data.columns)

            # Если есть столбцы с пропусками, выбираем первый
            missing_by_column = self.data.isnull().sum()
            columns_with_missing = missing_by_column[missing_by_column > 0].index.tolist()

            if columns_with_missing:
                self.ui.column_combo_missing.setCurrentText(columns_with_missing[0])

        except Exception as e:
            QMessageBox.critical(self, "Ошибка",
                                 f"Ошибка при загрузке страницы пропусков: {str(e)}")

    def update_missing_info(self):
        """Обновление информации о пропусках"""
        try:
            # Подсчитываем общую статистику
            total_rows = len(self.data)
            total_missing = self.data.isnull().sum().sum()
            rows_with_any_missing = self.data.isnull().any(axis=1).sum()
            rows_with_all_missing = self.data.isnull().all(axis=1).sum()

            # Подсчитываем по столбцам
            missing_by_column = self.data.isnull().sum()
            columns_with_missing = missing_by_column[missing_by_column > 0].index.tolist()

            # Формируем текст информации
            info_text = f"""Общая статистика пропусков:

Всего строк: {total_rows:,}
Всего пропусков: {total_missing:,}
Строк с пропусками: {rows_with_any_missing:,} ({rows_with_any_missing / total_rows * 100:.1f}%)
Строк полностью пустых: {rows_with_all_missing:,}

Пропуски по столбцам:
"""

            # Добавляем информацию по каждому столбцу с пропусками
            for col in columns_with_missing:
                missing_count = missing_by_column[col]
                percentage = missing_count / total_rows * 100
                info_text += f"\n{col}: {missing_count:,} ({percentage:.1f}%)"

            if not columns_with_missing:
                info_text += "\n\nВ данных нет пропусков!"

            self.ui.missing_info_text.setText(info_text)

            # Активируем/деактивируем кнопки
            selected_column = self.ui.column_combo_missing.currentText()
            if selected_column and selected_column in columns_with_missing:
                self.ui.clear_column_btn.setEnabled(True)
            else:
                self.ui.clear_column_btn.setEnabled(False)

            self.ui.clear_all_btn.setEnabled(total_missing > 0)

            # Обновляем статистику
            self.ui.stats_label_missing.setText(
                f"Найдено {total_missing} пропусков в {len(columns_with_missing)} столбцах"
            )

        except Exception as e:
            print(f"Ошибка при обновлении информации о пропусках: {str(e)}")

    def clear_column_missing(self):
        """Очистка пропусков в выбранном столбце"""
        try:
            column_name = self.ui.column_combo_missing.currentText()
            if not column_name:
                QMessageBox.warning(self, "Предупреждение", "Выберите столбец!")
                return

            column_data = self.data[column_name]
            missing_before = column_data.isnull().sum()

            if missing_before == 0:
                QMessageBox.information(self, "Информация",
                                        f"В столбце '{column_name}' нет пропусков")
                return

            # Подтверждение
            reply = QMessageBox.question(
                self, "Подтверждение очистки",
                f"Удалить {missing_before} пропусков из столбца '{column_name}'?\n\n"
                f"Это удалит строки с пропусками в этом столбце.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.No:
                return

            # Удаляем строки с пропусками в выбранном столбце
            self.data = self.data.dropna(subset=[column_name]).reset_index(drop=True)

            # Помечаем, что данные были изменены
            self.data_changed = True

            missing_after = self.data[column_name].isnull().sum()
            rows_removed = missing_before  # Так как мы удалили строки

            QMessageBox.information(self, "Успех",
                                    f"Пропуски в столбце '{column_name}' очищены!\n\n"
                                    f"Удалено строк: {rows_removed}\n"
                                    f"Осталось пропусков в столбце: {missing_after}")

            # Обновляем информацию
            self.update_missing_info()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при очистке столбца: {str(e)}")

    def clear_all_missing(self):
        """Очистка всех пропусков"""
        try:
            missing_before = self.data.isnull().sum().sum()

            if missing_before == 0:
                QMessageBox.information(self, "Информация", "В данных нет пропусков")
                return

            # Подтверждение
            reply = QMessageBox.question(
                self, "Подтверждение очистки",
                f"Удалить все {missing_before} пропусков из данных?\n\n"
                f"Это удалит все строки, содержащие хотя бы один пропуск.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.No:
                return

            # Сохраняем старый размер
            old_size = len(self.data)

            # Удаляем все строки с пропусками
            self.data = self.data.dropna().reset_index(drop=True)

            # Помечаем, что данные были изменены
            self.data_changed = True

            missing_after = self.data.isnull().sum().sum()
            rows_removed = old_size - len(self.data)

            QMessageBox.information(self, "Успех",
                                    f"Все пропуски очищены!\n\n"
                                    f"Было: {old_size} строк\n"
                                    f"Стало: {len(self.data)} строк\n"
                                    f"Удалено строк: {rows_removed}\n"
                                    f"Осталось пропусков: {missing_after}")

            # Обновляем информацию
            self.update_missing_info()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при очистке пропусков: {str(e)}")

    def on_close_button_clicked(self):
        """Обработчик нажатия кнопки 'Закрыть'"""
        current_index = self.ui.stackedWidget.currentIndex()
        max_index = self.ui.stackedWidget.count() - 1

        # Если мы на последней странице, сохраняем данные (как кнопка "Завершить")
        self.save_data_only()
        self.close_window()

    def save_data_only(self):
        """Сохранить данные без обновления состояния"""
        try:
            if self.filename.endswith('.csv'):
                if hasattr(self, 'current_delimiter'):
                    self.data.to_csv(self.file_path, index=False, sep=self.current_delimiter,
                                     encoding='utf-8')
                else:
                    self.data.to_csv(self.file_path, index=False, encoding='utf-8')
            elif self.filename.endswith('.json'):
                self.data.to_json(self.file_path, orient='records', indent=2, force_ascii=False)
        except Exception as e:
            print(f"Ошибка при сохранении данных: {str(e)}")

    def save_and_close_with_state_update(self):
        """Сохранить данные И обновить состояние в родительском окне"""
        try:
            # Сохраняем данные
            if self.filename.endswith('.csv'):
                if hasattr(self, 'current_delimiter'):
                    delimiter = self.current_delimiter
                    self.data.to_csv(self.file_path, index=False, sep=delimiter, encoding='utf-8')
                else:
                    delimiter = ','  # Используем запятую по умолчанию
                    self.data.to_csv(self.file_path, index=False, encoding='utf-8')
            elif self.filename.endswith('.json'):
                delimiter = None  # Для JSON нет разделителя
                self.data.to_json(self.file_path, orient='records', indent=2, force_ascii=False)

            # Обновляем состояние файла в родительском окне
            if self.parent_window:
                # Обновляем состояние preprocessing на True
                self.parent_window.update_file_state('preprocessing', completed=True)

                # Обновляем разделитель
                if hasattr(self, 'current_delimiter') and self.current_delimiter:
                    self.parent_window.update_file_separated(self.current_delimiter)

                # Обновляем кнопки в главном окне
                self.parent_window.update_analysis_buttons_state()

            QMessageBox.information(self, "Успех",
                                    f"Предобработка файла '{self.filename}' завершена!\n\n"
                                    f"Теперь доступны:\n"
                                    f"• Визуализация данных\n"
                                    f"• Моделирование\n\n"
                                    f" Данные сохранены.")
            self.close_window()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при сохранении данных: {str(e)}")

    def complete_preprocessing(self):
        """Завершение предобработки - сохраняет и обновляет состояние"""
        self.save_and_close_with_state_update()

    def close_window(self):
        """Закрытие окна без предупреждения"""
        # Удаляем временные файлы
        self.cleanup_temp_files()

        self.closed.emit()
        self.close()

    def cleanup_temp_files(self):
        """Очистка временных файлов"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"Не удалось удалить временный файл {temp_file}: {str(e)}")
        self.temp_files = []

    def closeEvent(self, event):
        """Обработчик закрытия окна системным крестиком"""
        # При закрытии крестиком просто сохраняем данные, не обновляем состояние

        self.save_data_only()

        # Очищаем временные файлы
        self.cleanup_temp_files()

        self.closed.emit()
        event.accept()
