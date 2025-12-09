import sys
import os
import json
import pandas as pd
import numpy as np
import tempfile
import shutil
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

# PyQt6 imports
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QListWidget, QListWidgetItem,
                             QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox,
                             QFrame, QStackedWidget, QCheckBox, QGridLayout,
                             QRadioButton, QMessageBox, QScrollArea,
                             QSizePolicy, QGroupBox, QAbstractItemView,
                             QFileDialog, QScrollBar, QApplication)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QRect, QTimer
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QWheelEvent

# Matplotlib imports
import matplotlib
matplotlib.use('QtAgg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import cm
import matplotlib.gridspec as gridspec

# Импортируем UI из сгенерированного файла
try:
    from form.visualization_window_ui import Ui_Visualization
except ImportError:
    print("Ошибка: Не удалось импортировать Ui_Visualization из form.visualization")
    print("Создаю простой UI для теста...")


    class Ui_Visualization:
        def setupUi(self, Visualization):
            Visualization.setObjectName("Visualization")
            Visualization.resize(800, 600)

            self.centralwidget = QWidget(Visualization)
            self.verticalLayout = QVBoxLayout(self.centralwidget)

            self.label = QLabel("Тестовый интерфейс", self.centralwidget)
            self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.verticalLayout.addWidget(self.label)

            self.btn_test = QPushButton("Тест", self.centralwidget)
            self.verticalLayout.addWidget(self.btn_test)

            Visualization.setCentralWidget(self.centralwidget)

        def retranslateUi(self, Visualization):
            pass


class ChartConfig:
    """Класс для хранения конфигурации графика для matplotlib/pandas"""

    def __init__(self, chart_type: str, title: str, data_config: Dict, styling: Dict,
                 fig_settings: Dict = None, created_at: str = None, id: str = None):
        self.chart_type = chart_type
        self.title = title
        self.data_config = data_config
        self.styling = styling
        self.created_at = created_at or datetime.now().isoformat()
        self.id = id or f"chart_{int(datetime.now().timestamp())}_{hash(title)}"
        self.image_path = None  # Путь к временному изображению

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'chart_type': self.chart_type,
            'title': self.title,
            'data_config': self.data_config,
            'styling': self.styling,
            'created_at': self.created_at
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ChartConfig':
        return cls(
            id=data.get('id'),
            chart_type=data['chart_type'],
            title=data['title'],
            data_config=data.get('data_config', {}),
            styling=data.get('styling', {}),
            created_at=data.get('created_at', datetime.now().isoformat())
        )


class ChartManager:
    """Менеджер для работы с конфигурациями графиков"""

    def __init__(self, app_data_dir: str = "data/app_data"):
        self.app_data_dir = app_data_dir
        os.makedirs(app_data_dir, exist_ok=True)

    def get_charts_file_path(self, dataset_name: str) -> str:
        base_name = os.path.splitext(dataset_name)[0]
        return os.path.join(self.app_data_dir, f"{base_name}_charts.json")

    def save_charts(self, dataset_name: str, charts: List[ChartConfig]):
        file_path = self.get_charts_file_path(dataset_name)
        charts_data = [chart.to_dict() for chart in charts]

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(charts_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Ошибка при сохранении графиков: {e}")
            return False

    def load_charts(self, dataset_name: str) -> List[ChartConfig]:
        file_path = self.get_charts_file_path(dataset_name)

        if not os.path.exists(file_path):
            return []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                charts_data = json.load(f)

            return [ChartConfig.from_dict(data) for data in charts_data]
        except Exception as e:
            print(f"Ошибка при загрузке графиков: {e}")
            return []


class UnifiedChartRenderer:
    """Рендерер, который создает отдельные окна matplotlib для каждого графика"""

    def __init__(self, data: pd.DataFrame, main_window=None, filename: str = ""):
        self.data = data
        self.main_window = main_window  # Ссылка на главное окно для скрытия/показа
        self.filename = filename  # Название файла для заголовка
        self.figures = []  # Список созданных фигур
        self.is_closing = False  # Флаг для предотвращения рекурсии

    def render_single_chart(self, chart: ChartConfig, show_window=True) -> Optional[Figure]:
        """Создает отдельную фигуру для одного графика"""
        try:
            chart_type = chart.chart_type
            data_config = chart.data_config
            styling = chart.styling

            if show_window:
                # Создаем фигуру через plt.figure
                fig = plt.figure(figsize=(10, 7), dpi=100)
                ax = fig.add_subplot(111)

                # Рендерим график
                self._render_chart_on_axes(ax, chart)

                # Устанавливаем заголовок
                ax.set_title(chart.title, fontsize=14, pad=12)

                # Настройки сетки
                if styling.get('show_grid', True):
                    ax.grid(True, alpha=0.3, linestyle='--')

                fig.tight_layout()

                # Устанавливаем заголовок окна с названием файла
                if self.filename:
                    # Убираем расширение файла для красивого отображения
                    base_name = os.path.splitext(self.filename)[0]
                    window_title = f"Дашборд: {base_name}"
                else:
                    window_title = "Дашборд"

                fig.canvas.manager.set_window_title(window_title)

                # Сохраняем ссылку на фигуру
                self.figures.append(fig)

                # Настраиваем обработчик закрытия окна
                def on_close(event):
                    try:
                        if fig in self.figures:
                            self.figures.remove(fig)
                        # Показываем главное окно при закрытии графика
                        if self.main_window and not self.is_closing:
                            self.main_window.show_main_window()
                    except Exception as e:
                        print(f"Ошибка в обработчике закрытия: {e}")

                fig.canvas.mpl_connect('close_event', on_close)

                # Показываем график неблокирующим образом
                plt.show(block=False)

                return fig
            else:
                # Для экспорта используем обычную фигуру
                fig = Figure(figsize=(10, 7), dpi=100)
                ax = fig.add_subplot(111)

                self._render_chart_on_axes(ax, chart)
                ax.set_title(chart.title, fontsize=14, pad=12)

                if styling.get('show_grid', True):
                    ax.grid(True, alpha=0.3, linestyle='--')

                fig.tight_layout()
                return fig

        except Exception as e:
            print(f"Ошибка при создании графика: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _render_multiple_charts_in_grid(self, charts: List[ChartConfig], rows: int, cols: int):
        """Создает одну фигуру с несколькими субграфиками в сетке"""
        try:
            # Адаптивная высота: больше строк = больше высота на график
            base_height_per_row = 4.0  # Базовая высота на строку

            if rows == 1:
                height_multiplier = 5.0  # Для одной строки делаем выше
            elif rows == 2:
                height_multiplier = 4.5
            elif rows == 3:
                height_multiplier = 4.0
            else:  # 4 строки
                height_multiplier = 3.8

            fig_width = cols * 5
            fig_height = rows * height_multiplier

            fig = plt.figure(figsize=(fig_width, fig_height), dpi=100)

            # Адаптивные отступы в зависимости от размера сетки
            if rows * cols <= 4:
                hspace, wspace = 0.5, 0.4
            elif rows * cols <= 9:
                hspace, wspace = 0.4, 0.3
            else:
                hspace, wspace = 0.35, 0.25

            plt.subplots_adjust(hspace=hspace, wspace=wspace)

            for i, chart in enumerate(charts):
                if i >= rows * cols:
                    break

                ax = fig.add_subplot(rows, cols, i + 1)
                self._render_chart_on_axes(ax, chart)

                # Адаптивный размер заголовка
                if rows * cols <= 4:
                    title_size = 10
                    title_pad = 10
                elif rows * cols <= 9:
                    title_size = 9
                    title_pad = 8
                else:
                    title_size = 8
                    title_pad = 6

                ax.set_title(chart.title, fontsize=title_size, pad=title_pad)

                if chart.styling.get('show_grid', True):
                    ax.grid(True, alpha=0.3, linestyle='--')

                # Адаптивный размер шрифта
                label_size = 8 if rows * cols <= 9 else 7
                tick_size = 7 if rows * cols <= 9 else 6

                ax.xaxis.label.set_size(label_size)
                ax.yaxis.label.set_size(label_size)
                ax.tick_params(axis='both', labelsize=tick_size)

                # Автоповорот для длинных подписей
                if len(ax.get_xticklabels()) > 0:
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=tick_size)

            # Пустые ячейки
            for i in range(len(charts), rows * cols):
                ax = fig.add_subplot(rows, cols, i + 1)
                ax.axis('off')
                ax.text(0.5, 0.5, 'Пусто',
                        ha='center', va='center',
                        transform=ax.transAxes,
                        fontsize=10, color='gray')

            fig.tight_layout(pad=3.0, h_pad=2.0 if rows > 1 else 1.5, w_pad=3.5)

            # Заголовок окна
            if self.filename:
                base_name = os.path.splitext(self.filename)[0]
                window_title = f"Дашборд: {base_name}"
            else:
                window_title = "Дашборд"

            fig.canvas.manager.set_window_title(window_title)
            plt.show(block=False)
            self.figures.append(fig)

            def on_close(event):
                try:
                    if fig in self.figures:
                        self.figures.remove(fig)
                    if self.main_window and not self.is_closing:
                        self.main_window.show_main_window()
                except Exception as e:
                    print(f"Ошибка в обработчике закрытия: {e}")

            fig.canvas.mpl_connect('close_event', on_close)

        except Exception as e:
            print(f"Ошибка при создании сетки графиков: {e}")
            import traceback
            traceback.print_exc()

    def render_all_charts(self, charts: List[ChartConfig], layout_config: Dict) -> bool:
        """Создает отдельные окна для всех графиков"""
        if not charts:
            return False

        # Закрываем предыдущие графики если есть
        self.close_all_charts()

        rows = layout_config['rows']
        cols = layout_config['cols']
        total_slots = rows * cols
        charts_to_render = charts[:total_slots]

        print(f"Создаем {len(charts_to_render)} графиков в компоновке {rows}x{cols}")

        # Скрываем главное окно
        if self.main_window:
            self.main_window.hide()

        # Рендерим каждый график в отдельном окне или в субграфиках одной фигуры
        if rows == 1 and cols == 1:
            # Для одного графика - отдельное окно
            if charts_to_render:
                self.render_single_chart(charts_to_render[0])
        else:
            # Для нескольких графиков - создаем одну фигуру с субграфиками
            self._render_multiple_charts_in_grid(charts_to_render, rows, cols)

        return True

    def _render_chart_on_axes(self, ax, chart_config: ChartConfig):
        """Рендерит конкретный график на заданных осях"""
        try:
            chart_type = chart_config.chart_type
            data_config = chart_config.data_config
            styling = chart_config.styling

            if chart_type == "Линейный график (plot)":
                self._render_line_chart(ax, data_config, styling)
            elif chart_type == "Столбчатая диаграмма (bar)":
                self._render_bar_chart(ax, data_config, styling)
            elif chart_type == "Круговая диаграмма (pie)":
                self._render_pie_chart(ax, data_config, styling)
            elif chart_type == "Гистограмма (hist)":
                self._render_histogram(ax, data_config, styling)
            elif chart_type == "Диаграмма рассеяния (scatter)":
                self._render_scatter(ax, data_config, styling)
            elif chart_type == "Box Plot (boxplot)":
                self._render_boxplot(ax, data_config, styling)
            elif chart_type == "Площадной график (area)":
                self._render_area_chart(ax, data_config, styling)
            elif chart_type == "График плотности (kde)":
                self._render_kde_chart(ax, data_config, styling)

        except Exception as e:
            ax.clear()
            error_msg = f"Ошибка: {str(e)[:50]}"
            ax.text(0.5, 0.5, error_msg,
                    ha='center', va='center',
                    transform=ax.transAxes,
                    color='red', fontsize=8)
            ax.set_title(f"Ошибка", color='red', fontsize=9)
            ax.axis('off')

    def _render_line_chart(self, ax, data_config: Dict, styling: Dict):
        x_col = data_config.get('x_column')
        y_cols = data_config.get('y_columns', [])

        if not x_col or not y_cols:
            ax.text(0.5, 0.5, "Не указаны данные",
                    ha='center', va='center', transform=ax.transAxes, color='red')
            return

        if x_col not in self.data.columns:
            ax.text(0.5, 0.5, f"Колонка X '{x_col}' не найдена",
                    ha='center', va='center', transform=ax.transAxes, color='red')
            return

        # Получаем доступные цвета для линий
        colors = plt.cm.Set2(np.linspace(0, 1, len(y_cols))) if len(y_cols) > 1 else ['blue']

        for i, y_col in enumerate(y_cols):
            if y_col not in self.data.columns:
                continue

            # Парсим параметры стиля
            line_style = self._parse_line_style(styling.get('line_style', 'solid (сплошная)'))
            line_width = styling.get('line_width', 2.0)
            markers = self._parse_marker(styling.get('markers', 'None (нет)'))
            markersize = styling.get('markersize', 6)
            alpha = styling.get('alpha', 1.0)

            x_data = self.data[x_col]
            y_data = self.data[y_col]

            mask = x_data.notna() & y_data.notna()
            x_clean = x_data[mask]
            y_clean = y_data[mask]

            if len(x_clean) > 0:
                df_plot = pd.DataFrame({
                    'x': x_clean.values,
                    'y': y_clean.values
                })

                # Сортируем по возрастанию X (как в временных рядах)
                df_plot = df_plot.sort_values('x')
                ax.plot(df_plot['x'].values, df_plot['y'].values,
                        linestyle=line_style,
                        linewidth=line_width,
                        marker=markers,
                        markersize=markersize,
                        alpha=alpha,
                        color=colors[i] if len(y_cols) > 1 else None,
                        label=y_col)

        if len(y_cols) > 1 and styling.get('show_legend', True):
            ax.legend(fontsize=7, loc='upper right')

        xlabel = styling.get('xlabel', x_col)
        ylabel = styling.get('ylabel', 'Значения')
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)

        # Настройка сетки
        if styling.get('show_grid', True):
            ax.grid(True, alpha=0.3, linestyle='--')

        # Настройка подписей осей
        if len(ax.get_xticklabels()) > 0:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)

    def _render_bar_chart(self, ax, data_config: Dict, styling: Dict):
        categories_col = data_config.get('categories_column')
        values_col = data_config.get('values_column')

        if categories_col not in self.data.columns or values_col not in self.data.columns:
            ax.text(0.5, 0.5, "Не указаны данные",
                    ha='center', va='center', transform=ax.transAxes, color='red')
            return

        # Парсим параметры стиля
        orientation = styling.get('orientation', 'vertical (вертикальная)')
        edgecolor = self._parse_color(styling.get('edgecolor', 'black (черный)'))
        edgewidth = styling.get('edgewidth', 1.0)
        alpha = styling.get('alpha', 0.8)
        width = styling.get('width', 0.8)

        # Получаем цвета для столбцов
        categories = self.data[categories_col].astype(str)
        values = self.data[values_col]

        # Создаем массив цветов
        colors = plt.cm.tab20c(np.linspace(0, 1, len(categories)))

        if 'вертикальная' in orientation.lower():
            # Вертикальная ориентация
            bars = ax.bar(categories, values,
                          color=colors,
                          edgecolor=edgecolor if edgecolor != 'none' else None,
                          linewidth=edgewidth,
                          alpha=alpha,
                          width=width)
            ax.set_xlabel(categories_col, fontsize=9)
            ax.set_ylabel(values_col, fontsize=9)

            # Добавляем значения на столбцы
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=7)

            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        else:
            # Горизонтальная ориентация
            bars = ax.barh(categories, values,
                           color=colors,
                           edgecolor=edgecolor if edgecolor != 'none' else None,
                           linewidth=edgewidth,
                           alpha=alpha,
                           height=width)
            ax.set_xlabel(values_col, fontsize=9)
            ax.set_ylabel(categories_col, fontsize=9)

            # Добавляем значения на столбцы
            for bar in bars:
                width_val = bar.get_width()
                ax.text(width_val, bar.get_y() + bar.get_height() / 2.,
                        f'{width_val:.1f}', ha='left', va='center', fontsize=7)

            plt.setp(ax.get_yticklabels(), fontsize=8)

    def _render_pie_chart(self, ax, data_config: Dict, styling: Dict):
        labels_col = data_config.get('labels_column')
        values_col = data_config.get('values_column')

        if labels_col not in self.data.columns or values_col not in self.data.columns:
            ax.text(0.5, 0.5, "Не указаны данные",
                    ha='center', va='center', transform=ax.transAxes, color='red')
            return

        # Парсим параметры стиля
        start_angle = styling.get('start_angle', 90)
        explode = styling.get('explode', 0.1)
        autopct = styling.get('autopct', 'Не показывать')
        show_shadow = styling.get('shadow', False)

        labels = self.data[labels_col].astype(str)
        values = self.data[values_col]

        # Берем только первые 8 значений для читаемости
        if len(values) > 8:
            values = values[:8]
            labels = labels[:8]

        # Создаем цвета
        colors = plt.cm.Set3(np.linspace(0, 1, len(values)))

        autopct_format = None
        if autopct != 'Не показывать':
            if '%1.1f%%' in autopct:
                autopct_format = '%1.1f%%'
            elif '%1.2f%%' in autopct:
                autopct_format = '%1.2f%%'
            elif '%d%%' in autopct:
                autopct_format = '%d%%'
            else:
                autopct_format = '%1.1f%%'

        if explode > 0:
            explode_values = [explode] + [0] * (len(values) - 1)
        else:
            explode_values = None

        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            autopct=autopct_format,
            colors=colors,
            startangle=start_angle,
            explode=explode_values,
            shadow=show_shadow,
            textprops={'fontsize': 8}
        )

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        # Добавляем легенду
        ax.legend(wedges, labels, title=labels_col,
                  loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
                  fontsize=7)

        ax.axis('equal')

    def _render_histogram(self, ax, data_config: Dict, styling: Dict):
        column = data_config.get('column')

        if column not in self.data.columns:
            ax.text(0.5, 0.5, "Не указаны данные",
                    ha='center', va='center', transform=ax.transAxes, color='red')
            return

        # Парсим параметры стиля
        bins = styling.get('bins', 'auto (автоматически)')
        hist_type = styling.get('hist_type', 'bar (столбчатая)')
        alpha = styling.get('alpha', 0.7)
        show_density = styling.get('density', False)
        show_cumulative = styling.get('cumulative', False)

        data = self.data[column].dropna()

        # Парсим количество bins
        if isinstance(bins, str) and 'auto' in bins.lower():
            bins = 'auto'
        else:
            try:
                bins = int(bins)
            except:
                bins = 10

        # Парсим тип гистограммы
        if 'bar' in hist_type.lower():
            hist_type_parsed = 'bar'
        elif 'step' in hist_type.lower():
            hist_type_parsed = 'step'
        elif 'stepfilled' in hist_type.lower():
            hist_type_parsed = 'stepfilled'
        else:
            hist_type_parsed = 'bar'

        # Выбираем цвет
        color = plt.cm.Blues(0.6)

        ax.hist(data, bins=bins, histtype=hist_type_parsed,
                color=color,
                alpha=alpha,
                density=show_density,
                cumulative=show_cumulative,
                edgecolor='black',
                linewidth=0.5)

        ax.set_xlabel(column, fontsize=9)
        ylabel = 'Плотность вероятности' if show_density else 'Частота'
        ax.set_ylabel(ylabel, fontsize=9)

        # Добавляем линии среднего и медианы
        ax.axvline(data.mean(), color='red', linestyle='dashed', linewidth=1,
                   label=f'Среднее: {data.mean():.2f}')
        ax.axvline(data.median(), color='green', linestyle='dashed', linewidth=1,
                   label=f'Медиана: {data.median():.2f}')

        if not show_cumulative:  # Легенду только для некумулятивной гистограммы
            ax.legend(fontsize=7)

    def _render_scatter(self, ax, data_config: Dict, styling: Dict):
        x_col = data_config.get('x_column')
        y_col = data_config.get('y_column')
        color_col = data_config.get('color_column', 'Нет')

        if x_col not in self.data.columns or y_col not in self.data.columns:
            ax.text(0.5, 0.5, "Не указаны данные",
                    ha='center', va='center', transform=ax.transAxes, color='red')
            return

        # Парсим параметры стиля
        point_size = styling.get('point_size', 50)
        point_alpha = styling.get('point_alpha', 0.6)
        marker = self._parse_marker(styling.get('marker', 'o (круг)'))
        show_regression = styling.get('regression', False)
        colormap_name = styling.get('colormap', 'viridis')

        x_data = self.data[x_col]
        y_data = self.data[y_col]

        mask = x_data.notna() & y_data.notna()
        x_clean = x_data[mask]
        y_clean = y_data[mask]

        # СОРТИРОВКА для линии регрессии (если она будет рисоваться)
        # Сохраняем индексы сортировки для использования в scatter
        if show_regression:
            # Сортируем данные для линии регрессии
            sorted_indices = np.argsort(x_clean.values)
            x_sorted_for_reg = x_clean.iloc[sorted_indices]
            y_sorted_for_reg = y_clean.iloc[sorted_indices]

        # Обработка цветового кодирования
        if color_col and color_col != "Нет" and color_col in self.data.columns:
            color_data = self.data[color_col][mask]

            # Если нужна сортировка для цветов (если цвета зависят от X)
            # Создаем DataFrame для сохранения соответствия
            if show_regression:
                # Сохраняем цвета в отсортированном порядке
                color_data_sorted = color_data.iloc[sorted_indices]

            try:
                # Пробуем преобразовать к числовому типу
                color_numeric = pd.to_numeric(color_data, errors='coerce')

                if color_numeric.notna().all():
                    # Числовые данные - используем colormap
                    scatter = ax.scatter(x_clean, y_clean,
                                         c=color_numeric,
                                         s=point_size,
                                         alpha=point_alpha,
                                         marker=marker,
                                         cmap=colormap_name,
                                         edgecolors='black',
                                         linewidth=0.5)

                    # Добавляем colorbar
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label(color_col, fontsize=8)
                    cbar.ax.tick_params(labelsize=7)

                else:
                    # Категориальные данные
                    categories = color_data.astype('category')
                    color_codes = categories.cat.codes

                    # Создаем цветовую карту для категорий
                    cmap = plt.cm.get_cmap(colormap_name)
                    num_categories = len(categories.cat.categories)

                    scatter = ax.scatter(x_clean, y_clean,
                                         c=color_codes,
                                         s=point_size,
                                         alpha=point_alpha,
                                         marker=marker,
                                         cmap=cmap,
                                         edgecolors='black',
                                         linewidth=0.5)

                    # Создаем кастомную легенду
                    handles = []
                    labels = []

                    for i, category in enumerate(categories.cat.categories):
                        color = cmap(i / max(1, num_categories - 1))
                        handles.append(plt.Line2D([0], [0],
                                                  marker='o',
                                                  color='w',
                                                  markerfacecolor=color,
                                                  markersize=8,
                                                  alpha=point_alpha,
                                                  markeredgecolor='black',
                                                  markeredgewidth=0.5))
                        labels.append(str(category))

                    ax.legend(handles, labels, title=color_col,
                              fontsize=7, loc='upper right')

            except Exception as e:
                print(f"Ошибка при обработке данных цвета: {e}")
                # В случае ошибки используем цвет по умолчанию
                ax.scatter(x_clean, y_clean,
                           s=point_size,
                           alpha=point_alpha,
                           marker=marker,
                           color='blue',
                           edgecolors='black',
                           linewidth=0.5)
        else:
            # Без цветового кодирования
            ax.scatter(x_clean, y_clean,
                       s=point_size,
                       alpha=point_alpha,
                       marker=marker,
                       color='blue',
                       edgecolors='black',
                       linewidth=0.5)

        # Линия регрессии - ИСПРАВЛЕННАЯ ЧАСТЬ
        if show_regression and len(x_clean) > 1:
            try:
                # Используем ОРИГИНАЛЬНЫЕ несортированные данные для расчета регрессии
                # Это важно, чтобы не исказить коэффициенты
                z = np.polyfit(x_clean, y_clean, 1)
                p = np.poly1d(z)

                # Для отрисовки линии используем СОРТИРОВАННЫЕ данные
                # чтобы линия была гладкой и не "прыгала"
                x_line = x_sorted_for_reg
                y_reg = p(x_line)

                ax.plot(x_line, y_reg, "r--", linewidth=2, label='Линия тренда')

                # Добавляем уравнение линии
                slope = z[0]
                intercept = z[1]
                eq_text = f"y = {slope:.2f}x + {intercept:.2f}"
                ax.text(0.05, 0.95, eq_text,
                        transform=ax.transAxes,
                        fontsize=8,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                ax.legend(fontsize=7)
            except Exception as e:
                print(f"Ошибка при построении линии регрессии: {e}")

        ax.set_xlabel(x_col, fontsize=9)
        ax.set_ylabel(y_col, fontsize=9)

        # Настройка сетки
        ax.grid(True, alpha=0.3, linestyle='--')

    def _render_boxplot(self, ax, data_config: Dict, styling: Dict):
        category_col = data_config.get('category_column', 'Нет (один бокс)')
        values_col = data_config.get('values_column')

        if values_col not in self.data.columns:
            ax.text(0.5, 0.5, "Не указаны данные",
                    ha='center', va='center', transform=ax.transAxes, color='red')
            return

        # Парсим параметры стиля
        orientation = styling.get('orientation', 'vertical (вертикальная)')
        show_points = styling.get('show_points', 'outliers (только выбросы)')
        show_notch = styling.get('notch', False)
        linewidth = styling.get('linewidth', 1.5)
        whis = styling.get('whis', 1.5)

        # Определяем, показывать ли выбросы
        showfliers = 'outliers' in show_points.lower() or 'all' in show_points.lower()

        # Определяем ориентацию
        vert = 'вертикальная' in orientation.lower()

        if category_col != "Нет (один бокс)" and category_col in self.data.columns:
            # Группировка по категориям
            data = []
            labels = []

            for category in sorted(self.data[category_col].astype(str).unique()):
                subset = self.data[self.data[category_col].astype(str) == category][values_col]
                if len(subset) > 0:
                    data.append(subset.dropna().values)
                    labels.append(str(category))

            # Цвета для boxplot
            colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
        else:
            # Один boxplot
            data = [self.data[values_col].dropna().values]
            labels = [values_col]
            colors = ['lightblue']

        # Создаем boxplot
        bp = ax.boxplot(data,
                        tick_labels=labels,
                        notch=show_notch,
                        showfliers=showfliers,
                        patch_artist=True,
                        boxprops=dict(linewidth=linewidth, color='black'),
                        whiskerprops=dict(linewidth=linewidth, color='black'),
                        capprops=dict(linewidth=linewidth, color='black'),
                        medianprops=dict(linewidth=linewidth, color='red'),
                        flierprops=dict(marker='o', markersize=4, alpha=0.5,
                                        markeredgecolor='black'),
                        whis=whis,
                        vert=vert)

        # Заполняем box'ы цветом
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Настройка осей в зависимости от ориентации
        if vert:
            ax.set_ylabel(values_col, fontsize=9)
            ax.set_xlabel(category_col if category_col != "Нет (один бокс)" else '', fontsize=9)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        else:
            ax.set_xlabel(values_col, fontsize=9)
            ax.set_ylabel(category_col if category_col != "Нет (один бокс)" else '', fontsize=9)
            plt.setp(ax.get_yticklabels(), fontsize=8)

        # Добавляем сетку
        ax.grid(True, alpha=0.3, linestyle='--', axis='y' if vert else 'x')

        # Добавляем точки данных если нужно
        if 'all' in show_points.lower():
            for i, d in enumerate(data):
                if vert:
                    # Для вертикальной ориентации
                    x = np.random.normal(i + 1, 0.04, size=len(d))
                    ax.plot(x, d, 'r.', alpha=0.5, markersize=3)
                else:
                    # Для горизонтальной ориентации
                    y = np.random.normal(i + 1, 0.04, size=len(d))
                    ax.plot(d, y, 'r.', alpha=0.5, markersize=3)

    def _render_area_chart(self, ax, data_config: Dict, styling: Dict):
        x_col = data_config.get('x_column')
        y_cols = data_config.get('y_columns', [])

        if x_col not in self.data.columns:
            ax.text(0.5, 0.5, "Не указаны данные",
                    ha='center', va='center', transform=ax.transAxes, color='red')
            return

        # Парсим параметры стиля
        alpha = styling.get('alpha', 0.5)
        colormap = styling.get('colormap', 'viridis')

        x_data = self.data[x_col]
        y_data_list = []
        valid_cols = []

        for y_col in y_cols:
            if y_col in self.data.columns:
                y_data = self.data[y_col]
                mask = x_data.notna() & y_data.notna()
                if len(x_data[mask]) > 0:
                    y_data_list.append(y_data[mask].values)
                    valid_cols.append(y_col)

        if y_data_list:
            try:
                cmap = plt.cm.get_cmap(colormap)
            except:
                cmap = plt.cm.viridis

            colors = cmap(np.linspace(0.2, 0.8, len(y_data_list)))

            ax.stackplot(x_data[x_data.notna()].values, *y_data_list,
                         labels=valid_cols, colors=colors, alpha=alpha)

            ax.set_xlabel(x_col, fontsize=9)
            ax.set_ylabel('Значения', fontsize=9)

            # Настройка сетки
            ax.grid(True, alpha=0.3, linestyle='--')

            if len(valid_cols) > 1:
                ax.legend(fontsize=7, loc='upper right')

            # Настройка формата дат если x_col это дата
            if pd.api.types.is_datetime64_any_dtype(x_data):
                ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        else:
            ax.text(0.5, 0.5, "Нет данных для отображения",
                    ha='center', va='center', transform=ax.transAxes, color='red')

    def _render_kde_chart(self, ax, data_config: Dict, styling: Dict):
        columns = data_config.get('columns', [])

        if not columns:
            ax.text(0.5, 0.5, "Не указаны столбцы",
                    ha='center', va='center', transform=ax.transAxes, color='red')
            return

        alpha = styling.get('alpha', 0.5)
        colormap = styling.get('colormap', 'viridis')
        show_fill = styling.get('fill', True)
        bandwidth = styling.get('bandwidth', None)

        try:
            from scipy import stats
        except ImportError:
            ax.text(0.5, 0.5, "Для KDE установите scipy",
                    ha='center', va='center', transform=ax.transAxes, color='red')
            return

        try:
            cmap = plt.cm.get_cmap(colormap)
        except:
            cmap = plt.cm.viridis

        colors = cmap(np.linspace(0, 1, len(columns)))

        for i, column in enumerate(columns):
            if column in self.data.columns:
                data = self.data[column].dropna()
                if len(data) > 1:
                    try:
                        kde = stats.gaussian_kde(data, bw_method=bandwidth)
                        x_range = np.linspace(data.min(), data.max(), 100)
                        y_kde = kde(x_range)

                        if show_fill:
                            ax.fill_between(x_range, y_kde, alpha=alpha * 0.7, color=colors[i])

                        ax.plot(x_range, y_kde, color=colors[i], linewidth=2, label=column)

                        # Добавляем вертикальную линию для среднего
                        mean_val = data.mean()
                        ax.axvline(mean_val, color=colors[i], linestyle='--', linewidth=1,
                                   alpha=0.7)

                    except Exception as e:
                        print(f"Ошибка при построении KDE для {column}: {e}")
                        continue

        ax.set_xlabel('Значения', fontsize=9)
        ax.set_ylabel('Плотность вероятности', fontsize=9)

        # Настройка сетки
        ax.grid(True, alpha=0.3, linestyle='--')

        if columns:
            ax.legend(fontsize=7)

        # Добавляем подпись
        ax.text(0.02, 0.98, 'KDE (Ядерная оценка плотности)',
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def _parse_line_style(self, style_str: str) -> str:
        """Парсит строку стиля линии"""
        style_mapping = {
            'solid (сплошная)': '-',
            'dashed (пунктирная)': '--',
            'dashdot (штрих-пунктир)': '-.',
            'dotted (точки)': ':'
        }
        return style_mapping.get(style_str, '-')

    def _parse_marker(self, marker_str: str) -> str:
        """Парсит строку маркера"""
        if 'None' in marker_str or 'нет' in marker_str.lower():
            return None

        marker_mapping = {
            'o (круг)': 'o',
            's (квадрат)': 's',
            '^ (треугольник)': '^',
            'D (ромб)': 'D',
            '+ (плюс)': '+',
            'x (крест)': 'x',
            'v (треугольник вниз)': 'v',
            '< (треугольник влево)': '<',
            '> (треугольник вправо)': '>',
            '8 (восьмиугольник)': '8'
        }

        # Извлекаем символ маркера из строки
        for key, value in marker_mapping.items():
            if key.startswith(marker_str.split(' ')[0]):
                return value
        return 'o'

    def _parse_color(self, color_str: str) -> str:
        """Парсит строку цвета"""
        if 'none' in color_str.lower() or 'без' in color_str.lower():
            return 'none'

        color_mapping = {
            'black (черный)': 'black',
            'darkgray': 'darkgray',
            'red': 'red',
            'blue': 'blue',
            'green': 'green'
        }
        return color_mapping.get(color_str, color_str.split(' ')[0])

    def close_all_charts(self):
        """Закрывает все открытые графики"""
        self.is_closing = True
        try:
            import matplotlib.pyplot as plt

            # Создаем копию списка чтобы избежать проблем с итерацией
            for fig in self.figures[:]:
                try:
                    plt.close(fig.number if hasattr(fig, 'number') else fig)
                except Exception as e:
                    print(f"Ошибка при закрытии графика: {e}")
                    try:
                        plt.close(fig)
                    except:
                        pass

            # Очищаем список
            self.figures.clear()

            # Закрываем все окна matplotlib
            try:
                plt.close('all')
            except:
                pass
        finally:
            self.is_closing = False

    def check_open_charts(self):
        """Проверяет, есть ли открытые графики"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib._pylab_helpers import Gcf

            # Проверяем, есть ли открытые фигуры
            fig_managers = Gcf.get_all_fig_managers()
            has_open_figures = len(fig_managers) > 0

            # Обновляем наш список фигур
            current_figures = [manager.canvas.figure for manager in fig_managers]
            self.figures = [fig for fig in self.figures if fig in current_figures]

            return has_open_figures
        except:
            return len(self.figures) > 0


class VisualizationWindow(QMainWindow):
    closed = pyqtSignal()

    def __init__(self, filename: str, parent=None):
        super().__init__(parent)

        self.filename = filename
        self.parent_window = parent
        self.data = None
        self.charts = []
        self.current_chart_index = -1
        self.layout_config = {'rows': 1, 'cols': 1}
        self.chart_manager = ChartManager()
        self.chart_renderer = None
        self.temp_dir = tempfile.mkdtemp(prefix="visualization_")

        # Флаги для управления состоянием
        self._closing_by_user = False  # Закрытие инициировано пользователем
        self.charts_are_open = False   # Графики matplotlib открыты
        self._ignore_close_events = False  # Игнорировать события закрытия

        # Маппинг типов графиков на страницы
        self.chart_type_to_page = {
            "Линейный график (plot)": 1,
            "Столбчатая диаграмма (bar)": 2,
            "Круговая диаграмма (pie)": 3,
            "Гистограмма (hist)": 4,
            "Диаграмма рассеяния (scatter)": 5,
            "Box Plot (boxplot)": 6,
            "Площадной график (area)": 1,
            "График плотности (kde)": 4
        }

        self.page_to_chart_type = {
            1: "Линейный график (plot)",
            2: "Столбчатая диаграмма (bar)",
            3: "Круговая диаграмма (pie)",
            4: "Гистограмма (hist)",
            5: "Диаграмма рассеяния (scatter)",
            6: "Box Plot (boxplot)"
        }

        self.ui = Ui_Visualization()
        self.ui.setupUi(self)

        self.setup_connections()
        self.load_data()
        self.setup_initial_state()
        self.load_saved_charts()

        # Инициализация рендерера после загрузки данных
        if self.data is not None:
            self.chart_renderer = UnifiedChartRenderer(self.data, self, self.filename)

        # Таймер для проверки закрытия графиков
        self.chart_check_timer = QTimer()
        self.chart_check_timer.timeout.connect(self.check_matplotlib_windows)
        self.chart_check_timer.start(500)

    def setup_connections(self):
        # Основные навигационные кнопки
        self.ui.btn_prev.clicked.connect(self.go_to_previous_page)
        self.ui.btn_next.clicked.connect(self.go_to_next_page)
        self.ui.btn_finish.clicked.connect(self.finish_visualization)

        # Кнопки на страницах графиков
        self.ui.btn_configure_chart.clicked.connect(self.configure_chart)
        self.ui.btn_line_back.clicked.connect(lambda: self.go_to_page(0))
        self.ui.btn_line_save.clicked.connect(lambda: self.save_chart(1))
        self.ui.btn_bar_back.clicked.connect(lambda: self.go_to_page(0))
        self.ui.btn_bar_save.clicked.connect(lambda: self.save_chart(2))
        self.ui.btn_pie_back.clicked.connect(lambda: self.go_to_page(0))
        self.ui.btn_pie_save.clicked.connect(lambda: self.save_chart(3))
        self.ui.btn_hist_back.clicked.connect(lambda: self.go_to_page(0))
        self.ui.btn_hist_save.clicked.connect(lambda: self.save_chart(4))
        self.ui.btn_scatter_back.clicked.connect(lambda: self.go_to_page(0))
        self.ui.btn_scatter_save.clicked.connect(lambda: self.save_chart(5))
        self.ui.btn_box_back.clicked.connect(lambda: self.go_to_page(0))
        self.ui.btn_box_save.clicked.connect(lambda: self.save_chart(6))

        # Кнопки на странице компоновки
        self.ui.btn_back_to_setup.clicked.connect(lambda: self.go_to_page(0))
        self.ui.btn_export_figure.clicked.connect(self.export_figure)

        # Кнопки управления списком графиков
        self.ui.btn_edit_chart.clicked.connect(self.edit_chart)
        self.ui.btn_remove_chart.clicked.connect(self.remove_chart)

        # События выбора в списке графиков
        self.ui.charts_listwidget.itemSelectionChanged.connect(self.on_chart_selected)

        # Радиокнопки компоновки
        self.ui.radio_1x1.toggled.connect(lambda: self.on_layout_radio_toggled(1, 1))
        self.ui.radio_2x2.toggled.connect(lambda: self.on_layout_radio_toggled(2, 2))
        self.ui.radio_3x3.toggled.connect(lambda: self.on_layout_radio_toggled(3, 3))
        self.ui.radio_2x3.toggled.connect(lambda: self.on_layout_radio_toggled(2, 3))
        self.ui.radio_3x2.toggled.connect(lambda: self.on_layout_radio_toggled(3, 2))
        self.ui.radio_4x4.toggled.connect(lambda: self.on_layout_radio_toggled(4, 4))
        self.ui.radio_self.toggled.connect(self.on_custom_layout_toggled)

        # Спинбоксы для кастомного лейаута
        self.ui.custom_rows_spin.valueChanged.connect(self.update_custom_layout)
        self.ui.custom_cols_spin.valueChanged.connect(self.update_custom_layout)

        # Простой тест если это тестовый UI
        if hasattr(self.ui, 'btn_test'):
            self.ui.btn_test.clicked.connect(self.test_function)

    def test_function(self):
        QMessageBox.information(self, "Тест", "Программа запущена успешно!")

    def setup_initial_state(self):
        # Получаем базовое имя файла для заголовка
        base_name = os.path.splitext(self.filename)[0]
        self.setWindowTitle(f"DataLite - Визуализация: {base_name}")

        if hasattr(self.ui, 'stackedWidget'):
            self.max_page_index = self.ui.stackedWidget.count() - 1
            print(f"Всего страниц в stackedWidget: {self.max_page_index + 1}")
            self.ui.stackedWidget.setCurrentIndex(0)

        self.update_navigation_buttons()
        self.populate_column_comboboxes()
        self.update_data_info()

    def load_data(self):
        """Загрузка данных с определением сепаратора и правильных типов столбцов"""
        try:
            file_path = f"data/storage/{self.filename}"
            print(f"Попытка загрузить файл: {file_path}")

            if not os.path.exists(file_path):
                print("Файл не найден, создаю тестовые данные...")
                self.data = pd.DataFrame({
                    'Дата': pd.date_range(start='2023-01-01', periods=10, freq='D'),
                    'Продажи': [100, 120, 130, 110, 150, 140, 160, 170, 180, 190],
                    'Прибыль': [20, 25, 30, 22, 35, 33, 38, 40, 42, 45],
                    'Товар': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
                    'Количество': [10, 15, 12, 8, 18, 14, 16, 20, 22, 24],
                    'Цена': [10.0, 8.0, 10.8, 13.7, 8.3, 10.0, 9.9, 8.5, 8.2, 7.9],
                    'Возраст': [25, 30, 35, 28, 40, 32, 45, 38, 29, 42],
                    'Зарплата': [50000, 55000, 60000, 52000, 70000, 58000, 75000, 65000, 53000,
                                 72000]
                })
                print(f"Созданы тестовые данные: {len(self.data)} строк")
            else:
                if self.filename.endswith('.csv'):
                    # Определяем сепаратор автоматически
                    separator = self.detect_csv_separator(file_path)
                    print(f"Определен сепаратор: '{separator}'")

                    # Загружаем с автоматическим определением типов данных
                    self.data = pd.read_csv(file_path, sep=separator)

                    # Пробуем автоматически определить типы данных
                    self.convert_data_types()

                elif self.filename.endswith('.json'):
                    self.data = pd.read_json(file_path)
                elif self.filename.endswith('.xlsx'):
                    self.data = pd.read_excel(file_path)
                else:
                    QMessageBox.critical(self, "Ошибка", "Неподдерживаемый формат файла")
                    return

            print(f"Данные загружены: {len(self.data)} строк, {len(self.data.columns)} колонок")
            print(f"Типы данных столбцов:")
            for col in self.data.columns:
                print(f"  {col}: {self.data[col].dtype}")

        except Exception as e:
            print(f"Ошибка при загрузке данных: {e}")
            import traceback
            traceback.print_exc()
            self.data = pd.DataFrame({
                'Дата': pd.date_range(start='2023-01-01', periods=5, freq='D'),
                'Продажи': [100, 120, 130, 110, 150],
                'Прибыль': [20, 25, 30, 22, 35],
                'Товар': ['A', 'B', 'A', 'C', 'B'],
                'Количество': [10, 15, 12, 8, 18]
            })
            print("Использованы тестовые данные из-за ошибки")

    def detect_csv_separator(self, file_path, sample_lines=10):
        """Автоматическое определение разделителя в CSV файле"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = []
                for _ in range(sample_lines):
                    line = f.readline()
                    if not line:
                        break
                    lines.append(line)

            if not lines:
                return ','

            # Подсчитываем количество различных разделителей
            separators = [',', ';', '\t', '|']
            separator_counts = {}

            for sep in separators:
                counts = []
                for line in lines:
                    # Игнорируем строки, которые выглядят как даты или содержат мало символов
                    if len(line.strip()) > 10:
                        count = line.count(sep)
                        if count > 0:
                            counts.append(count)

                if counts:
                    # Берем медианное значение для устойчивости к выбросам
                    separator_counts[sep] = np.median(counts) if counts else 0

            # Выбираем разделитель с максимальным количеством вхождений
            if separator_counts:
                best_separator = max(separator_counts, key=separator_counts.get)
                if separator_counts[best_separator] > 0:
                    return best_separator

            # Если не нашли разделитель, пробуем определить по первой строке
            first_line = lines[0].strip()
            if '\t' in first_line and first_line.count('\t') > first_line.count(','):
                return '\t'
            elif ';' in first_line and first_line.count(';') > first_line.count(','):
                return ';'
            elif '|' in first_line:
                return '|'

            return ','

        except Exception as e:
            print(f"Ошибка при определении разделителя: {e}")
            return ','

    def convert_data_types(self):
        """Автоматическое преобразование типов данных столбцов"""
        if self.data is None or self.data.empty:
            return

        print("Преобразование типов данных столбцов...")

        for column in self.data.columns:
            try:
                # Сохраняем исходные значения для отладки
                original_type = self.data[column].dtype
                original_sample = self.data[column].head(5).tolist() if len(self.data) > 0 else []

                # Пробуем преобразовать к числовым типам
                if not pd.api.types.is_numeric_dtype(self.data[column]):
                    # Пробуем преобразовать к числам
                    numeric_data = pd.to_numeric(self.data[column], errors='coerce')

                    # Если удалось преобразовать хотя бы 70% значений
                    non_null_count = numeric_data.notna().sum()
                    total_count = len(numeric_data)

                    if total_count > 0 and (non_null_count / total_count) > 0.7:
                        self.data[column] = numeric_data
                        new_type = self.data[column].dtype
                        print(f"  {column}: {original_type} -> {new_type} "
                              f"(успешно преобразовано: {non_null_count}/{total_count})")
                    else:
                        # Пробуем преобразовать к дате/времени
                        try:
                            date_data = pd.to_datetime(self.data[column], errors='coerce',
                                                       dayfirst=True)
                            non_null_dates = date_data.notna().sum()

                            if total_count > 0 and (non_null_dates / total_count) > 0.7:
                                self.data[column] = date_data
                                new_type = self.data[column].dtype
                                print(f"  {column}: {original_type} -> {new_type} "
                                      f"(даты: {non_null_dates}/{total_count})")
                            else:
                                # Если не удалось преобразовать, оставляем как строки
                                # Но пробуем убрать лишние пробелы
                                if self.data[column].dtype == 'object':
                                    self.data[column] = self.data[column].astype(str).str.strip()
                                print(
                                    f"  {column}: {original_type} -> остается как {self.data[column].dtype}")
                        except:
                            if self.data[column].dtype == 'object':
                                self.data[column] = self.data[column].astype(str).str.strip()
                            print(
                                f"  {column}: {original_type} -> остается как {self.data[column].dtype}")
                else:
                    # Уже числовой тип, но проверяем на целочисленность
                    if pd.api.types.is_float_dtype(self.data[column]):
                        # Проверяем, можно ли преобразовать к целым числам
                        if self.data[column].notna().all():
                            int_data = self.data[column].astype(int)
                            if (int_data == self.data[column]).all():
                                self.data[column] = int_data
                                print(
                                    f"  {column}: {original_type} -> {self.data[column].dtype} (целые числа)")

                    print(f"  {column}: уже числовой ({self.data[column].dtype})")

            except Exception as e:
                print(f"  Ошибка при преобразовании столбца {column}: {e}")
                # В случае ошибки оставляем как есть
                continue

        print("Преобразование типов данных завершено")

    def get_suitable_columns(self, chart_type, required_numeric=True):
        """Возвращает подходящие столбцы для типа графика с учетом типов данных"""
        if self.data is None or self.data.empty:
            return []

        columns = []
        for column in self.data.columns:
            try:
                # Проверяем тип данных столбца
                column_dtype = self.data[column].dtype
                column_has_numeric = False

                # Проверяем, содержит ли столбец числовые данные
                if pd.api.types.is_numeric_dtype(column_dtype):
                    column_has_numeric = True
                elif column_dtype == 'object' or column_dtype == 'string':
                    # Пробуем проверить, можно ли преобразовать к числам
                    try:
                        numeric_test = pd.to_numeric(self.data[column], errors='coerce')
                        if numeric_test.notna().any():
                            column_has_numeric = True
                    except:
                        pass

                # Для разных типов графиков разные требования
                if chart_type in ["Гистограмма (hist)", "Box Plot (boxplot)",
                                  "График плотности (kde)", "Диаграмма рассеяния (scatter)"]:
                    if column_has_numeric:
                        columns.append(column)
                elif chart_type in ["Круговая диаграмма (pie)", "Столбчатая диаграмма (bar)"]:
                    # Для pie и bar можно использовать любые столбцы
                    # Но для значений нужны числовые, для категорий - любые
                    columns.append(column)
                elif chart_type in ["Линейный график (plot)", "Площадной график (area)"]:
                    columns.append(column)
            except Exception as e:
                print(f"Ошибка при проверке столбца {column}: {e}")
                columns.append(column)

        return columns

    def update_data_info(self):
        if self.data is not None and not self.data.empty:
            base_name = os.path.splitext(self.filename)[0]
            info_text = f"Дашборд: {base_name} | Данные: {len(self.data)} строк, {len(self.data.columns)} колонки"
            if hasattr(self.ui, 'label_data_info'):
                self.ui.label_data_info.setText(info_text)
            print(info_text)

    def populate_column_comboboxes(self):
        if self.data is None or self.data.empty:
            return

        all_columns = self.data.columns.tolist()
        print(f"Все колонки: {all_columns}")

        for chart_type in self.chart_type_to_page.keys():
            suitable_columns = self.get_suitable_columns(chart_type)

            if chart_type == "Линейный график (plot)":
                if hasattr(self.ui, 'line_x_combo'):
                    self.ui.line_x_combo.clear()
                    self.ui.line_x_combo.addItems([""] + suitable_columns)
                if hasattr(self.ui, 'line_y_list'):
                    self.ui.line_y_list.clear()
                    for column in suitable_columns:
                        item = QListWidgetItem(column)
                        item.setCheckState(Qt.CheckState.Unchecked)
                        self.ui.line_y_list.addItem(item)
            elif chart_type == "Столбчатая диаграмма (bar)":
                if hasattr(self.ui, 'bar_categories_combo'):
                    self.ui.bar_categories_combo.clear()
                    self.ui.bar_categories_combo.addItems([""] + all_columns)
                if hasattr(self.ui, 'bar_values_combo'):
                    self.ui.bar_values_combo.clear()
                    numeric_cols = self.get_suitable_columns("Столбчатая диаграмма (bar)")
                    self.ui.bar_values_combo.addItems([""] + numeric_cols)
            elif chart_type == "Круговая диаграмма (pie)":
                if hasattr(self.ui, 'pie_labels_combo'):
                    self.ui.pie_labels_combo.clear()
                    self.ui.pie_labels_combo.addItems([""] + all_columns)
                if hasattr(self.ui, 'pie_values_combo'):
                    self.ui.pie_values_combo.clear()
                    numeric_cols = self.get_suitable_columns("Круговая диаграмма (pie)")
                    self.ui.pie_values_combo.addItems([""] + numeric_cols)
            elif chart_type == "Гистограмма (hist)":
                if hasattr(self.ui, 'hist_column_combo'):
                    self.ui.hist_column_combo.clear()
                    numeric_cols = self.get_suitable_columns("Гистограмма (hist)")
                    self.ui.hist_column_combo.addItems([""] + numeric_cols)
            elif chart_type == "Диаграмма рассеяния (scatter)":
                if hasattr(self.ui, 'scatter_x_combo'):
                    self.ui.scatter_x_combo.clear()
                    numeric_cols = self.get_suitable_columns("Диаграмма рассеяния (scatter)")
                    self.ui.scatter_x_combo.addItems([""] + numeric_cols)
                if hasattr(self.ui, 'scatter_y_combo'):
                    self.ui.scatter_y_combo.clear()
                    self.ui.scatter_y_combo.addItems([""] + numeric_cols)
                if hasattr(self.ui, 'scatter_color_combo'):
                    self.ui.scatter_color_combo.clear()
                    self.ui.scatter_color_combo.addItems(["Нет"] + all_columns)
            elif chart_type == "Box Plot (boxplot)":
                if hasattr(self.ui, 'box_category_combo'):
                    self.ui.box_category_combo.clear()
                    self.ui.box_category_combo.addItems(["Нет (один бокс)"] + all_columns)
                if hasattr(self.ui, 'box_values_combo'):
                    self.ui.box_values_combo.clear()
                    numeric_cols = self.get_suitable_columns("Box Plot (boxplot)")
                    self.ui.box_values_combo.addItems([""] + numeric_cols)

    def go_to_previous_page(self):
        if hasattr(self.ui, 'stackedWidget'):
            current_index = self.ui.stackedWidget.currentIndex()
            if 0 < current_index <= 7:
                self.ui.stackedWidget.setCurrentIndex(0)
            elif current_index > 0:
                self.ui.stackedWidget.setCurrentIndex(current_index - 1)
            self.update_navigation_buttons()

    def go_to_next_page(self):
        if hasattr(self.ui, 'stackedWidget'):
            current_index = self.ui.stackedWidget.currentIndex()
            max_index = self.ui.stackedWidget.count() - 1

            if current_index == 0:
                if not self.charts:
                    QMessageBox.warning(self, "Внимание", "Сначала создайте хотя бы один график!")
                    return
                self.go_to_page(7)
            elif 7 <= current_index < max_index:
                self.ui.stackedWidget.setCurrentIndex(current_index + 1)
                self.update_navigation_buttons()

    def go_to_page(self, page_index):
        """Переход на конкретную страницу"""
        if hasattr(self.ui, 'stackedWidget'):
            if 0 <= page_index < self.ui.stackedWidget.count():
                self.ui.stackedWidget.setCurrentIndex(page_index)
                self.update_navigation_buttons()
                print(f"Перешли на страницу {page_index}")
            else:
                print(f"Ошибка: страницы {page_index} не существует")

    def update_navigation_buttons(self):
        if hasattr(self.ui, 'stackedWidget'):
            current_index = self.ui.stackedWidget.currentIndex()
            print(f"Текущая страница: {current_index}")

            if hasattr(self.ui, 'btn_prev'):
                self.ui.btn_prev.setEnabled(current_index > 0)

            if hasattr(self.ui, 'btn_next'):
                if current_index == 0:
                    self.ui.btn_next.setText("К компоновке")
                    self.ui.btn_next.clicked.disconnect()
                    self.ui.btn_next.clicked.connect(self.go_to_next_page)
                    self.ui.btn_next.setEnabled(True)
                elif current_index == 7:
                    self.ui.btn_next.setText("Показать графики")
                    self.ui.btn_next.clicked.disconnect()
                    self.ui.btn_next.clicked.connect(self.generate_layout)
                    self.ui.btn_next.setEnabled(len(self.charts) > 0)
                else:
                    self.ui.btn_next.setEnabled(False)
                    self.ui.btn_next.setText("Далее")

            if hasattr(self.ui, 'btn_finish'):
                self.ui.btn_finish.setEnabled(current_index == 0)

    def configure_chart(self):
        """Настройка параметров графика - переход на страницу конкретного типа графика"""
        chart_type = self.ui.chart_type_combo.currentText()

        if not chart_type:
            QMessageBox.warning(self, "Внимание", "Выберите тип графика!")
            return

        title = self.ui.chart_title_edit.text().strip()
        if not title:
            QMessageBox.warning(self, "Внимание", "Введите название графика!")
            return

        page_index = self.chart_type_to_page.get(chart_type, 1)

        if page_index >= self.ui.stackedWidget.count():
            QMessageBox.warning(self, "Ошибка",
                                f"Страница настройки для типа '{chart_type}' не найдена")
            return

        self.go_to_page(page_index)

    def save_chart(self, page_type):
        """Сохранение конфигурации графика"""
        try:
            chart_type = ""
            data_config = {}
            styling = {}

            title = self.ui.chart_title_edit.text().strip()
            if not title:
                QMessageBox.warning(self, "Внимание", "Введите название графика!")
                return

            if page_type == 1:
                chart_type = "Линейный график (plot)"

                x_col = self.ui.line_x_combo.currentText()
                if not x_col:
                    QMessageBox.warning(self, "Внимание", "Выберите столбец для оси X!")
                    return

                y_cols = []
                for i in range(self.ui.line_y_list.count()):
                    item = self.ui.line_y_list.item(i)
                    if item.checkState() == Qt.CheckState.Checked:
                        y_cols.append(item.text())

                if not y_cols:
                    QMessageBox.warning(self, "Внимание",
                                        "Выберите хотя бы один столбец для оси Y!")
                    return

                data_config = {
                    'x_column': x_col,
                    'y_columns': y_cols
                }

                styling = {
                    'line_style': self.ui.line_style_combo.currentText(),
                    'line_width': self.ui.line_width_spin.value(),
                    'markers': self.ui.markers_combo.currentText(),
                    'markersize': self.ui.markersize_spin.value(),
                    'alpha': self.ui.line_alpha_spin.value(),
                    'show_grid': self.ui.grid_checkbox.isChecked(),
                    'show_legend': True,
                    'xlabel': self.ui.xlabel_edit.text() or x_col,
                    'ylabel': self.ui.ylabel_edit.text() or 'Значения'
                }

            elif page_type == 2:
                chart_type = "Столбчатая диаграмма (bar)"

                categories_col = self.ui.bar_categories_combo.currentText()
                values_col = self.ui.bar_values_combo.currentText()

                if not categories_col or not values_col:
                    QMessageBox.warning(self, "Внимание",
                                        "Выберите столбцы для категорий и значений!")
                    return

                data_config = {
                    'categories_column': categories_col,
                    'values_column': values_col
                }

                styling = {
                    'orientation': self.ui.bar_orientation_combo.currentText(),
                    'edgecolor': self.ui.bar_edgecolor_combo.currentText(),
                    'edgewidth': self.ui.bar_edgewidth_spin.value(),
                    'alpha': self.ui.bar_alpha_spin.value(),
                    'width': self.ui.bar_width_spin.value()
                }

            elif page_type == 3:
                chart_type = "Круговая диаграмма (pie)"

                labels_col = self.ui.pie_labels_combo.currentText()
                values_col = self.ui.pie_values_combo.currentText()

                if not labels_col or not values_col:
                    QMessageBox.warning(self, "Внимание", "Выберите столбцы для меток и значений!")
                    return

                data_config = {
                    'labels_column': labels_col,
                    'values_column': values_col
                }

                styling = {
                    'start_angle': self.ui.pie_start_angle_spin.value(),
                    'explode': self.ui.pie_explode_spin.value(),
                    'autopct': self.ui.pie_autopct_combo.currentText(),
                    'shadow': self.ui.pie_shadow_checkbox.isChecked(),
                }

            elif page_type == 4:
                chart_type = "Гистограмма (hist)"

                column = self.ui.hist_column_combo.currentText()
                if not column:
                    QMessageBox.warning(self, "Внимание", "Выберите столбец для гистограммы!")
                    return

                data_config = {
                    'column': column
                }

                styling = {
                    'bins': self.ui.bins_combo.currentText(),
                    'hist_type': self.ui.hist_type_combo.currentText(),
                    'alpha': self.ui.hist_alpha_spin.value(),
                    'density': self.ui.hist_density_checkbox.isChecked(),
                    'cumulative': self.ui.hist_cumulative_checkbox.isChecked()
                }

            elif page_type == 5:
                chart_type = "Диаграмма рассеяния (scatter)"

                x_col = self.ui.scatter_x_combo.currentText()
                y_col = self.ui.scatter_y_combo.currentText()

                if not x_col or not y_col:
                    QMessageBox.warning(self, "Внимание", "Выберите столбцы для осей X и Y!")
                    return

                data_config = {
                    'x_column': x_col,
                    'y_column': y_col
                }

                color_col = self.ui.scatter_color_combo.currentText()
                if color_col and color_col != "Нет":
                    data_config['color_column'] = color_col

                styling = {
                    'point_size': self.ui.point_size_spin.value(),
                    'point_alpha': self.ui.point_alpha_spin.value(),
                    'marker': self.ui.scatter_marker_combo.currentText(),
                    'regression': self.ui.regression_checkbox.isChecked(),
                    'colormap': self.ui.scatter_colormap_combo.currentText()
                }

            elif page_type == 6:
                chart_type = "Box Plot (boxplot)"

                values_col = self.ui.box_values_combo.currentText()
                if not values_col:
                    QMessageBox.warning(self, "Внимание", "Выберите столбец значений!")
                    return

                data_config = {
                    'values_column': values_col
                }

                category_col = self.ui.box_category_combo.currentText()
                if category_col and category_col != "Нет (один бокс)":
                    data_config['category_column'] = category_col

                styling = {
                    'orientation': self.ui.box_orientation_combo.currentText(),
                    'show_points': self.ui.box_show_points_combo.currentText(),
                    'notch': self.ui.box_notch_checkbox.isChecked(),
                    'linewidth': self.ui.box_linewidth_spin.value(),
                    'whis': self.ui.box_whis_spin.value()
                }

            chart_config = ChartConfig(
                chart_type=chart_type,
                title=title,
                data_config=data_config,
                styling=styling
            )

            self.charts.append(chart_config)
            self.update_charts_list()
            self.chart_manager.save_charts(self.filename, self.charts)
            self.go_to_page(0)

            QMessageBox.information(self, "Успех", f"График '{title}' сохранен!")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при сохранении графика: {str(e)}")
            print(f"Ошибка: {e}")
            import traceback
            traceback.print_exc()

    def update_charts_list(self):
        """Обновление списка графиков в QListWidget"""
        if hasattr(self.ui, 'charts_listwidget'):
            self.ui.charts_listwidget.clear()

            for i, chart in enumerate(self.charts):
                item_text = f"{i + 1}. {chart.title} ({chart.chart_type})"
                item = QListWidgetItem(item_text)
                item.setData(Qt.ItemDataRole.UserRole, i)
                self.ui.charts_listwidget.addItem(item)

    def on_chart_selected(self):
        """Обработка выбора графика в списке"""
        selected_items = self.ui.charts_listwidget.selectedItems()
        if selected_items:
            self.ui.btn_edit_chart.setEnabled(True)
            self.ui.btn_remove_chart.setEnabled(True)
        else:
            self.ui.btn_edit_chart.setEnabled(False)
            self.ui.btn_remove_chart.setEnabled(False)

    def edit_chart(self):
        """Редактирование выбранного графика"""
        selected_items = self.ui.charts_listwidget.selectedItems()
        if not selected_items:
            return

        item = selected_items[0]
        chart_index = item.data(Qt.ItemDataRole.UserRole)

        if 0 <= chart_index < len(self.charts):
            chart = self.charts[chart_index]

            self.ui.chart_type_combo.setCurrentText(chart.chart_type)
            self.ui.chart_title_edit.setText(chart.title)

            chart_type = chart.chart_type
            page_index = self.chart_type_to_page.get(chart_type, 1)
            self.go_to_page(page_index)

            if chart_type == "Линейный график (plot)":
                self._fill_line_chart_fields(chart)
            elif chart_type == "Столбчатая диаграмма (bar)":
                self._fill_bar_chart_fields(chart)
            elif chart_type == "Круговая диаграмма (pie)":
                self._fill_pie_chart_fields(chart)
            elif chart_type == "Гистограмма (hist)":
                self._fill_histogram_fields(chart)
            elif chart_type == "Диаграмма рассеяния (scatter)":
                self._fill_scatter_fields(chart)
            elif chart_type == "Box Plot (boxplot)":
                self._fill_boxplot_fields(chart)

            del self.charts[chart_index]
            self.update_charts_list()

            QMessageBox.information(self, "Редактирование",
                                    f"График '{chart.title}' готов к редактированию. Настройте параметры и сохраните.")

    def _fill_line_chart_fields(self, chart):
        data_config = chart.data_config
        styling = chart.styling

        x_col = data_config.get('x_column', '')
        y_cols = data_config.get('y_columns', [])

        self.ui.line_x_combo.setCurrentText(x_col)

        for i in range(self.ui.line_y_list.count()):
            item = self.ui.line_y_list.item(i)
            item.setCheckState(
                Qt.CheckState.Checked if item.text() in y_cols else Qt.CheckState.Unchecked)

        self.ui.line_style_combo.setCurrentText(styling.get('line_style', 'solid (сплошная)'))
        self.ui.line_width_spin.setValue(styling.get('line_width', 2.0))
        self.ui.markers_combo.setCurrentText(styling.get('markers', 'None (нет)'))
        self.ui.markersize_spin.setValue(styling.get('markersize', 6))
        self.ui.line_alpha_spin.setValue(styling.get('alpha', 1.0))
        self.ui.grid_checkbox.setChecked(styling.get('show_grid', True))
        self.ui.xlabel_edit.setText(styling.get('xlabel', ''))
        self.ui.ylabel_edit.setText(styling.get('ylabel', ''))

    def _fill_bar_chart_fields(self, chart):
        data_config = chart.data_config
        styling = chart.styling

        self.ui.bar_categories_combo.setCurrentText(data_config.get('categories_column', ''))
        self.ui.bar_values_combo.setCurrentText(data_config.get('values_column', ''))

        self.ui.bar_orientation_combo.setCurrentText(
            styling.get('orientation', 'vertical (вертикальная)'))
        self.ui.bar_edgecolor_combo.setCurrentText(styling.get('edgecolor', 'black (черный)'))
        self.ui.bar_edgewidth_spin.setValue(styling.get('edgewidth', 1.0))
        self.ui.bar_alpha_spin.setValue(styling.get('alpha', 0.8))
        self.ui.bar_width_spin.setValue(styling.get('width', 0.8))

    def _fill_pie_chart_fields(self, chart):
        data_config = chart.data_config
        styling = chart.styling

        self.ui.pie_labels_combo.setCurrentText(data_config.get('labels_column', ''))
        self.ui.pie_values_combo.setCurrentText(data_config.get('values_column', ''))

        self.ui.pie_start_angle_spin.setValue(styling.get('start_angle', 90))
        self.ui.pie_explode_spin.setValue(styling.get('explode', 0.1))
        self.ui.pie_autopct_combo.setCurrentText(styling.get('autopct', 'Не показывать'))
        self.ui.pie_shadow_checkbox.setChecked(styling.get('shadow', False))

    def _fill_histogram_fields(self, chart):
        data_config = chart.data_config
        styling = chart.styling

        self.ui.hist_column_combo.setCurrentText(data_config.get('column', ''))

        self.ui.bins_combo.setCurrentText(styling.get('bins', 'auto (автоматически)'))
        self.ui.hist_type_combo.setCurrentText(styling.get('hist_type', 'bar (столбчатая)'))
        self.ui.hist_alpha_spin.setValue(styling.get('alpha', 0.7))
        self.ui.hist_density_checkbox.setChecked(styling.get('density', False))
        self.ui.hist_cumulative_checkbox.setChecked(styling.get('cumulative', False))

    def _fill_scatter_fields(self, chart):
        data_config = chart.data_config
        styling = chart.styling

        self.ui.scatter_x_combo.setCurrentText(data_config.get('x_column', ''))
        self.ui.scatter_y_combo.setCurrentText(data_config.get('y_column', ''))

        color_col = data_config.get('color_column', '')
        self.ui.scatter_color_combo.setCurrentText(color_col if color_col else "Нет")

        self.ui.point_size_spin.setValue(styling.get('point_size', 50))
        self.ui.point_alpha_spin.setValue(styling.get('point_alpha', 0.6))
        self.ui.scatter_marker_combo.setCurrentText(styling.get('marker', 'o (круг)'))
        self.ui.regression_checkbox.setChecked(styling.get('regression', False))
        self.ui.scatter_colormap_combo.setCurrentText(styling.get('colormap', 'viridis'))

    def _fill_boxplot_fields(self, chart):
        data_config = chart.data_config
        styling = chart.styling

        self.ui.box_values_combo.setCurrentText(data_config.get('values_column', ''))

        category_col = data_config.get('category_column', '')
        self.ui.box_category_combo.setCurrentText(
            category_col if category_col else "Нет (один бокс)")

        self.ui.box_orientation_combo.setCurrentText(
            styling.get('orientation', 'vertical (вертикальная)'))
        self.ui.box_show_points_combo.setCurrentText(
            styling.get('show_points', 'outliers (только выбросы)'))
        self.ui.box_notch_checkbox.setChecked(styling.get('notch', False))
        self.ui.box_linewidth_spin.setValue(styling.get('linewidth', 1.5))
        self.ui.box_whis_spin.setValue(styling.get('whis', 1.5))

    def remove_chart(self):
        """Удаление выбранного графика"""
        selected_items = self.ui.charts_listwidget.selectedItems()
        if not selected_items:
            return

        item = selected_items[0]
        chart_index = item.data(Qt.ItemDataRole.UserRole)

        if 0 <= chart_index < len(self.charts):
            chart = self.charts[chart_index]

            reply = QMessageBox.question(self, "Подтверждение",
                                         f"Вы уверены, что хотите удалить график '{chart.title}'?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

            if reply == QMessageBox.StandardButton.Yes:
                del self.charts[chart_index]
                self.update_charts_list()
                self.chart_manager.save_charts(self.filename, self.charts)

                self.ui.btn_edit_chart.setEnabled(False)
                self.ui.btn_remove_chart.setEnabled(False)

    def on_layout_radio_toggled(self, rows, cols):
        """Обработка выбора стандартной компоновки"""
        if self.sender().isChecked():
            self.layout_config = {'rows': rows, 'cols': cols}
            self.ui.custom_rows_spin.setEnabled(False)
            self.ui.custom_cols_spin.setEnabled(False)

            if hasattr(self.ui, 'current_layout_label'):
                total_charts = rows * cols
                self.ui.current_layout_label.setText(
                    f"Компоновка: {rows} × {cols} | Графиков: {total_charts}")

    def on_custom_layout_toggled(self, checked):
        """Обработка выбора кастомной компоновки"""
        if checked:
            self.ui.custom_rows_spin.setEnabled(True)
            self.ui.custom_cols_spin.setEnabled(True)
            self.update_custom_layout()

    def update_custom_layout(self):
        """Обновление кастомного лейаута"""
        if self.ui.radio_self.isChecked():
            rows = self.ui.custom_rows_spin.value()
            cols = self.ui.custom_cols_spin.value()
            self.layout_config = {'rows': rows, 'cols': cols}

            if hasattr(self.ui, 'current_layout_label'):
                total_charts = rows * cols
                self.ui.current_layout_label.setText(
                    f"Компоновка: {rows} × {cols} | Графиков: {total_charts}")

    def generate_layout(self):
        """Генерация графика(ов) в отдельном окне matplotlib"""
        print("Нажата кнопка 'Построить все графики'")

        if not self.charts:
            QMessageBox.warning(self, "Внимание", "Нет сохраненных графиков для отображения!")
            return

        if not self.chart_renderer:
            self.chart_renderer = UnifiedChartRenderer(self.data, self, self.filename)

        try:
            # Закрываем предыдущие графики если есть
            self.chart_renderer.close_all_charts()

            # Скрываем главное окно визуализации
            self.hide()
            self.charts_are_open = True  # Устанавливаем флаг, что графики открыты

            # Рендерим графики в отдельном окне matplotlib
            success = self.chart_renderer.render_all_charts(self.charts, self.layout_config)

            if success:
                base_name = os.path.splitext(self.filename)[0]
                QMessageBox.information(
                    self,
                    "Успех",
                    f"Дашборд '{base_name}' открыт в отдельном окне!\n\n"
                    f"Закройте окно дашборда чтобы вернуться к редактору."
                )
            else:
                # Если не удалось создать графики, показываем окно обратно
                self.charts_are_open = False
                self.show()
                QMessageBox.warning(self, "Ошибка", "Не удалось создать графики!")

        except Exception as e:
            self.charts_are_open = False
            self.show()
            QMessageBox.critical(self, "Ошибка", f"Ошибка при построении графиков: {str(e)}")
            print(f"Ошибка: {e}")
            import traceback
            traceback.print_exc()

    def check_matplotlib_windows(self):
        """Проверяет, есть ли открытые окна matplotlib"""
        try:
            # Если окно закрывается пользователем - не проверяем
            if self._closing_by_user:
                return

            if not self.chart_renderer:
                return

            # Проверяем, есть ли открытые графики
            has_open_charts = self.chart_renderer.check_open_charts()

            print(
                f"Проверка matplotlib окон: has_open_charts={has_open_charts}, charts_are_open={self.charts_are_open}")

            # Если графики были открыты, но сейчас закрыты
            if self.charts_are_open and not has_open_charts:
                print("Все графики закрыты, возвращаемся в окно визуализации...")
                self.charts_are_open = False

                # Устанавливаем флаг игнорирования событий закрытия
                self._ignore_close_events = True

                # Показываем окно визуализации
                self.show_main_window()

                # Сбрасываем флаг через короткое время
                QTimer.singleShot(100, self.reset_ignore_flag)

                # Показываем сообщение пользователю
                QMessageBox.information(
                    self,
                    "Возврат к редактору",
                    "Все графики закрыты. Вы вернулись в редактор визуализации."
                )

        except Exception as e:
            print(f"Ошибка при проверке окон matplotlib: {e}")

    def reset_ignore_flag(self):
        """Сбрасывает флаг игнорирования событий закрытия"""
        self._ignore_close_events = False
        print("Флаг игнорирования событий закрытия сброшен")

    def show_main_window(self):
        """Показывает главное окно и активирует его"""
        try:
            if not self.isVisible():
                self.show()
                self.raise_()
                self.activateWindow()
                # Даем фокус окну
                QApplication.processEvents()
                print("Окно визуализации показано после закрытия графиков")
        except Exception as e:
            print(f"Ошибка при показе окна визуализации: {e}")

    def finish_visualization(self):
        """Завершение работы с визуализацией (по нажатию кнопки 'Завершить')"""
        print("Нажата кнопка 'Завершить работу' - безопасное закрытие")

        # Устанавливаем флаг, что закрытие инициировано пользователем
        self._closing_by_user = True

        # Останавливаем таймер проверки
        if hasattr(self, 'chart_check_timer'):
            self.chart_check_timer.stop()
            print("Таймер остановлен")

        # Закрываем все графики matplotlib
        self.close_all_charts_and_show()
        print("Все графики закрыты")

        # Сохраняем графики
        if self.charts:
            success = self.chart_manager.save_charts(self.filename, self.charts)
            print(f"Графики сохранены: {success}")

        # Закрываем окно визуализации
        print("Закрываем окно визуализации...")
        self.closed.emit()  # Испускаем сигнал перед закрытием
        self.close()  # Вызываем закрытие окна

    def close_all_charts_and_show(self):
        """Принудительно закрывает все графики и показывает окно"""
        if self.chart_renderer:
            try:
                self.chart_renderer.close_all_charts()
                self.charts_are_open = False
            except Exception as e:
                print(f"Ошибка при закрытии графиков: {e}")

        # Показываем окно если оно скрыто
        if not self.isVisible():
            self.show_main_window()

    def closeEvent(self, event):
        """Обработчик закрытия окна (при нажатии на крестик или Alt+F4)"""
        print(
            f"closeEvent окна визуализации: _closing_by_user={self._closing_by_user}, _ignore_close_events={self._ignore_close_events}")

        # Если установлен флаг игнорирования - игнорируем событие
        if self._ignore_close_events:
            print("closeEvent: игнорируем событие (вызвано закрытием matplotlib окон)")
            event.ignore()
            return

        # Если уже закрывается пользователем (например, по кнопке) - принимаем событие
        if self._closing_by_user:
            print("closeEvent: окно уже закрывается пользователем, принимаем событие")
            event.accept()
            return

        # Если это системное закрытие (крестик, Alt+F4)
        print("closeEvent: системное закрытие окна визуализации (крестик/Alt+F4)")

        # Устанавливаем флаг закрытия пользователем
        self._closing_by_user = True

        # Останавливаем таймер
        if hasattr(self, 'chart_check_timer'):
            self.chart_check_timer.stop()
            print("Таймер остановлен в closeEvent")

        # Закрываем все графики matplotlib
        if self.chart_renderer:
            try:
                self.chart_renderer.close_all_charts()
                print("Графики закрыты в closeEvent")
            except Exception as e:
                print(f"Ошибка при закрытии графиков в closeEvent: {e}")

        # Сохраняем графики
        if self.charts:
            try:
                self.chart_manager.save_charts(self.filename, self.charts)
                print("Графики сохранены в closeEvent")
            except Exception as e:
                print(f"Ошибка при сохранении графиков: {e}")

        # Очистка временных файлов
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                print(f"Очищена временная директория: {self.temp_dir}")
            except Exception as e:
                print(f"Ошибка при очистке временной директории: {e}")

        # Устанавливаем флаг в родительском окне (если есть)
        if self.parent_window and hasattr(self.parent_window, '_processing_matplotlib_close'):
            self.parent_window._processing_matplotlib_close = True
            print("Установлен флаг _processing_matplotlib_close в родительском окне")

        # Испускаем сигнал о закрытии
        print("Испускаем сигнал closed")
        self.closed.emit()

        # Принимаем событие закрытия
        event.accept()
        print("Событие closeEvent принято")

        # Сбрасываем флаг в родительском окне через небольшое время
        if self.parent_window and hasattr(self.parent_window, '_processing_matplotlib_close'):
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(100, self.reset_parent_flag)

    def reset_parent_flag(self):
        """Сбрасывает флаг в родительском окне"""
        if self.parent_window and hasattr(self.parent_window, '_processing_matplotlib_close'):
            self.parent_window._processing_matplotlib_close = False
            print("Сброшен флаг _processing_matplotlib_close в родительском окне")

    def export_figure(self):
        """Экспорт выбранного графика в файл"""
        selected_items = self.ui.charts_listwidget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Внимание", "Выберите график для экспорта!")
            return

        item = selected_items[0]
        chart_index = item.data(Qt.ItemDataRole.UserRole)

        if 0 <= chart_index < len(self.charts):
            chart = self.charts[chart_index]

            # Создаем имя файла по умолчанию с названием датасета
            base_name = os.path.splitext(self.filename)[0]
            default_name = f"Дашборд_{base_name}_{chart.title}".replace(' ', '_')

            file_path, _ = QFileDialog.getSaveFileName(
                self,
                f"Сохранить график: {chart.title}",
                default_name,
                "PNG Files (*.png);;JPEG Files (*.jpg);;PDF Files (*.pdf);;SVG Files (*.svg)"
            )

            if file_path:
                try:
                    # Создаем временную фигуру для экспорта
                    if not self.chart_renderer:
                        self.chart_renderer = UnifiedChartRenderer(self.data, self, self.filename)

                    fig = self.chart_renderer.render_single_chart(chart, show_window=False)
                    if fig:
                        fig.savefig(file_path, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        QMessageBox.information(self, "Успех", f"График сохранен в {file_path}!")
                    else:
                        QMessageBox.warning(self, "Ошибка",
                                            "Не удалось создать график для экспорта!")

                except Exception as e:
                    QMessageBox.critical(self, "Ошибка", f"Ошибка при сохранении графика: {str(e)}")
                    print(f"Ошибка экспорта: {e}")

    def load_saved_charts(self):
        """Загрузка сохраненных графиков"""
        self.charts = self.chart_manager.load_charts(self.filename)
        if self.charts:
            print(f"Загружено {len(self.charts)} сохраненных графиков")
            self.update_charts_list()


# Для тестирования модуля
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Тестовый файл
    test_filename = "test_data.csv"

    # Создаем тестовые данные если их нет
    os.makedirs("data/storage", exist_ok=True)
    test_data = pd.DataFrame({
        'Дата': pd.date_range(start='2023-01-01', periods=10, freq='D'),
        'Продажи': [100, 120, 130, 110, 150, 140, 160, 170, 180, 190],
        'Прибыль': [20, 25, 30, 22, 35, 33, 38, 40, 42, 45],
        'Товар': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
        'Количество': [10, 15, 12, 8, 18, 14, 16, 20, 22, 24],
        'Цена': [10.0, 8.0, 10.8, 13.7, 8.3, 10.0, 9.9, 8.5, 8.2, 7.9]
    })
    test_data.to_csv(f"data/storage/{test_filename}", index=False)

    window = VisualizationWindow(test_filename)
    window.show()

    sys.exit(app.exec())
